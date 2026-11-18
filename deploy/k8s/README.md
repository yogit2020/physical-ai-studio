# Deploy Physical AI Studio with remote Pi0.5 and the Qwen safety monitor on Kubernetes

Three workloads, one image each:

| Workload | Image | Serves | GPU |
|---|---|---|---|
| `physical-ai-studio` | `ghcr.io/yogit2020/physical-ai-studio-xpu` (this branch) | UI + backend, port 7860 | none (uses the servers) |
| `pi05-policy` | `ghcr.io/yogit2020/pi05-policy-server-xpu` (Studio image + policy server) | Pi0.5 PyTorch export, `GET /info`, `POST /predict`, port 18767 | 1 Intel GPU |
| `qwen-safety` | `ghcr.io/yogit2020/qwen-safety-xpu` (vLLM 0.32 XPU + settings) | Qwen3.8-27B AWQ as `qwen-safety`, OpenAI API, port 8000 | 1 Intel GPU |

```
Browser ──7860──► physical-ai-studio ──18767──► pi05-policy   (actions, ~90 ms per chunk on a B70)
                         └────────8000────────► qwen-safety   (SAFE / UNSAFE_HAND / UNSAFE_OBJECT, ~0.5 s)
```

Layout:

```
deploy/
  docker/Dockerfile.pi05-policy-server   Studio image + remote_policy_server.py as entrypoint
  docker/Dockerfile.qwen-safety          vLLM XPU + qwen-safety-entrypoint.sh
  k8s/base/                              portable manifests (GPUs via DRA, models downloaded into PVCs)
  k8s/overlays/devcloud/                 Intel DevCloud B70 host: fixed cards, models already on disk
  k8s/components/robot-node/             optional: Studio on the node with the arm and cameras
```

## 1. Requirements

- Kubernetes 1.34+ with the `resource.k8s.io/v1` API (Dynamic Resource Allocation).
- The [Intel GPU resource driver](https://github.com/intel/intel-resource-drivers-for-kubernetes),
  which provides DeviceClass `gpu.intel.com`. Check: `kubectl get deviceclass gpu.intel.com`.
- Two free Intel GPUs with at least 24 GB each (tested on Arc Pro B70, 32 GB). Qwen takes ~26 GB, Pi0.5 ~9 GB.
- A default StorageClass for the PVCs (~110 GB in total), or the devcloud overlay's host paths.
- A Hugging Face token with access to the gated `google/paligemma-3b-pt-224` (Pi0.5's tokenizer).
- The host GID of `render` (`getent group render`). The base uses 990; change
  `supplementalGroups` in `base/pi05-policy.yaml` if yours differs.

## 2. Images

Prebuilt images are on `ghcr.io/yogit2020`. To build your own from a checkout of this branch (repo root):

```bash
TAG=$(git rev-parse --short=8 HEAD)
REG=ghcr.io/<owner>

docker buildx build --load -f application/docker/Dockerfile --build-context libs=library \
  --target physical-ai-studio-xpu --build-arg GIT_SHA=$TAG -t $REG/physical-ai-studio-xpu:$TAG .
docker build -f deploy/docker/Dockerfile.pi05-policy-server \
  --build-arg STUDIO_IMAGE=$REG/physical-ai-studio-xpu:$TAG -t $REG/pi05-policy-server-xpu:$TAG .
docker build -f deploy/docker/Dockerfile.qwen-safety -t $REG/qwen-safety-xpu:vllm-v0.32.0 .

docker push $REG/physical-ai-studio-xpu:$TAG
docker push $REG/pi05-policy-server-xpu:$TAG
docker push $REG/qwen-safety-xpu:vllm-v0.32.0
```

Then point the manifests at them: in `k8s/base/kustomization.yaml` set `newName` / `newTag` under `images:`,
or run `cd deploy/k8s/base && kustomize edit set image physical-ai-studio=$REG/physical-ai-studio-xpu:$TAG ...`.

Private packages need a pull secret in the namespace:

```bash
kubectl -n physical-ai-studio create secret docker-registry ghcr --docker-server=ghcr.io \
  --docker-username=<user> --docker-password=<token with read:packages>
kubectl -n physical-ai-studio patch serviceaccount default -p '{"imagePullSecrets": [{"name": "ghcr"}]}'
```

On a single node without a registry, load the images into containerd instead:
`docker save <image> | sudo ctr -n k8s.io images import -`.

## 3. Deploy (any cluster)

```bash
kubectl create namespace physical-ai-studio
kubectl -n physical-ai-studio create secret generic hf-token --from-literal=token=hf_xxx
kubectl apply -k deploy/k8s/base
```

On first start the init containers download the models into their PVCs (~20 GB for Qwen, ~8 GB for
Pi0.5); later restarts skip it. Then Pi0.5 loads and compiles (~5 min) and Qwen loads (~3 min):

```bash
kubectl -n physical-ai-studio get pods -w
kubectl -n physical-ai-studio logs deploy/pi05-policy -f     # "warmed up in ...", "serving on 0.0.0.0:18767"
kubectl -n physical-ai-studio logs deploy/qwen-safety -f     # "Application startup complete"
```

All three are `Ready` when their probes pass (`/api/health`, `/info`, `/v1/models`).

### Open the UI

```bash
kubectl -n physical-ai-studio port-forward svc/physical-ai-studio 7860:7860
```

and browse to <http://localhost:7860>. For a permanent address, put an Ingress or a `NodePort` /
`LoadBalancer` Service in front of `physical-ai-studio:7860` (the UI uses websockets).

### Robot and cameras

Studio drives the arm and reads the cameras over USB, so it must run on the machine they are plugged
into. Label that node and add the `robot-node` component (privileged, like the Docker Compose default):

```bash
kubectl label node <node> physical-ai-studio/robot=true
```

```yaml
# deploy/k8s/overlays/<yours>/kustomization.yaml
resources: [../../base]
components: [../../components/robot-node]
```

Edit the calibration host path in `components/robot-node/kustomization.yaml` to where the arms'
LeRobot calibration lives on that node. Without the component, Studio runs but sees no robot or camera.

## 4. Deploy on the Intel DevCloud B70 host

Both B70s on that host are already allocated to other workloads in Kubernetes, so the overlay gives each
server a fixed card with an admin-access claim (as the host's GPU monitor does) and uses the models
already on disk:

| Server | Card | PCI | Models |
|---|---|---|---|
| `qwen-safety` | `card1` / `renderD128` | `0000:18:00.0` | `/home/devcloud/yogini/qwen-safety/models` |
| `pi05-policy` | `card2` / `renderD129` | `0000:54:00.0` | `/home/devcloud/yogini/pi05-210base-65ft/exports`, tokenizer in `/home/devcloud/yogini/bench/pg_cache` |

```bash
kubectl apply -k deploy/k8s/overlays/devcloud
```

The Docker containers that use the same cards (`yogini-qwen-safety`, `yogini-pi05-torch`,
`yogini-pi05-server`) must be stopped first: the GPUs do not have the memory for both.

**Tested 2026-11-18** (images `18d8c6ec` and `vllm-v0.32.0`, Kubernetes 1.34, Intel GPU DRA driver):

| Check | Result |
|---|---|
| Scheduling | all 3 pods on the node; both claims allocated with admin access to the intended cards |
| GPU isolation | `qwen-safety` sees only `card1`/`renderD128`, `pi05-policy` only `card2`/`renderD129` |
| Start-up | Studio Ready in < 20 s; Pi0.5 load 101 s + compile warm-up 176 s (`--warmup-synthetic`); Qwen Ready in < 5 min |
| Pi0.5 from the Studio pod | `/info` fingerprint `c8af1928…` (same weights as before); `(50, 6)` actions, **84 ms** on the GPU, **90 ms** round trip (median of 22) |
| Qwen from the Studio pod | model `qwen-safety`; one-word check `SAFE`, **240 ms** median (first call 1.6 s) |
| Studio | `/api/health` healthy; server preset `pi05-policy:18767`; Test connection OK (12 ms); `SAFETY_MONITOR_ENDPOINT` and the scene prompt set |

Not tested there: a robot (none on that host), model import, and a real run with the safety monitor.

## 5. Use it

1. **Import the model in Studio.** Studio checks that the server runs the same model as the one you
   select, by a weights fingerprint, so Studio needs its own copy of the export. Create a project in
   the UI, then import its dataset and the model from Hugging Face with the CLI in the pod:

   ```bash
   S="kubectl -n physical-ai-studio exec -it deploy/physical-ai-studio -- physicalai-studio"
   $S hf dataset import muniker/so101-sort-colored-blocks-65-dataset --project-id <project-uuid>
   $S hf model import muniker/pi05-so101-sort-210base-65ft --project-id <project-uuid> --dataset-id <dataset-uuid>
   ```

   Add `--dry-run` first to check without downloading; `--help` lists the options. A model trained in
   this Studio is already there.
2. **Run model:** pick the model, backend **PyTorch**, device **remote**. The server is preset to
   `pi05-policy:18767` (from `base/remote_inference.json`); **Test connection** shows its latency and
   whether it runs the same model. Change it there if needed; Studio keeps it in its storage.
3. **Safety:** in the run-model panel, **Enable safety monitor**, **Disable safety monitor**,
   **Acknowledge** (clears a stop; nothing restarts on its own) and **Safety stop**. Every follower
   without its own monitor settings uses `SAFETY_MONITOR_ENDPOINT` (`http://qwen-safety:8000/v1`) and
   the scene prompt in `base/safety-scene-prompt.txt`. Describe your own work cell there: the prompt
   decides what counts as a hazard. Capture a home pose first, so a stop can send the arm home.

## 6. Check and troubleshoot

```bash
NS="-n physical-ai-studio"
kubectl $NS exec deploy/physical-ai-studio -- python -c \
  "import urllib.request as u; print(u.urlopen('http://pi05-policy:18767/info').read().decode())"
kubectl $NS exec deploy/physical-ai-studio -- python -c \
  "import urllib.request as u; print(u.urlopen('http://qwen-safety:8000/v1/models').read().decode()[:200])"
```

| Symptom | Cause |
|---|---|
| Pod `Pending`, event about resource claims | No free GPU matches the claim (`kubectl get resourceclaims -A`) |
| `pi05-policy` restarts during start-up | The startup probe allows 10 min; a cold Intel compile cache can take longer: raise `failureThreshold` |
| `Permission denied` on `/dev/dri/renderD*` | `supplementalGroups` is not the host's `render` GID |
| `fetch-model` fails with 401/403 | The `hf-token` secret is missing or has no access to `google/paligemma-3b-pt-224` |
| Test connection: "different model" | Studio's model and the server's export are not the same weights |
| Monitor on but health `off` | No endpoint: check `SAFETY_MONITOR_ENDPOINT` and that `qwen-safety` is Ready |

## 7. Remove

```bash
kubectl delete -k deploy/k8s/base          # or overlays/devcloud
```

This deletes the PVCs too, with Studio's database and the downloaded models (`local-path` reclaims with
`Delete`). Back up `/app/storage` first if you want to keep projects and datasets.
