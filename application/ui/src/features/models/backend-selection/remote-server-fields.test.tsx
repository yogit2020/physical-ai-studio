import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { getDefaultInferenceDevice } from './backend-selection';
import { RemoteServerFields } from './remote-server-fields';

const remoteDevice = {
    backend: 'torch' as const,
    device: 'REMOTE',
    type: 'xpu' as const,
    name: 'Remote server (set host and port)',
    memory: null,
    index: null,
};

describe('RemoteServerFields', () => {
    it('pre-fills the saved server and tests it against the model', async () => {
        const user = userEvent.setup();
        let checked: unknown;
        server.use(
            http.get('/api/system/remote-inference', () =>
                HttpResponse.json({ servers: { torch: { host: '172.18.0.1', port: 18767 } } })
            ),
            http.post('/api/system/remote-inference/check', async ({ request }) => {
                checked = await request.json();
                return HttpResponse.json({
                    ok: true,
                    url: 'http://172.18.0.1:18767',
                    latency_ms: 9.2,
                    backend: 'torch',
                    device: 'xpu',
                    same_model: true,
                });
            })
        );
        render(<RemoteServerFields backend='torch' modelId='model-1' />);

        await waitFor(() => expect(screen.getByRole('textbox', { name: /host/i })).toHaveValue('172.18.0.1'));
        await user.click(screen.getByRole('button', { name: 'Test connection' }));

        expect(await screen.findByText(/Connected \(torch on xpu, 9 ms\), same model/)).toBeInTheDocument();
        expect(checked).toEqual({ host: '172.18.0.1', port: 18767, model_id: 'model-1', backend: 'torch' });
    });

    it('saves a changed server before testing it, and reports a server that runs another model', async () => {
        const user = userEvent.setup();
        let saved: unknown;
        server.use(
            http.get('/api/system/remote-inference', () => HttpResponse.json({ servers: {} })),
            http.put('/api/system/remote-inference/{backend}', async ({ request, params }) => {
                saved = { backend: params.backend, ...((await request.json()) as object) };
                return HttpResponse.json({ servers: { openvino: { host: '10.0.0.5', port: 18765 } } });
            }),
            http.post('/api/system/remote-inference/check', () =>
                HttpResponse.json({
                    ok: true,
                    url: 'http://10.0.0.5:18765',
                    latency_ms: 12,
                    backend: 'openvino',
                    device: 'GPU',
                    same_model: false,
                })
            )
        );
        render(<RemoteServerFields backend='openvino' modelId='model-1' />);

        const test = screen.getByRole('button', { name: /test connection/i });
        expect(test).toBeDisabled();
        await user.type(screen.getByRole('textbox', { name: /host/i }), '10.0.0.5');
        await user.type(screen.getByRole('textbox', { name: /port/i }), '18765');
        await user.tab(); // the port field commits its number on blur
        await user.click(screen.getByRole('button', { name: 'Save and test connection' }));

        expect(await screen.findByText(/runs another model than this one/)).toBeInTheDocument();
        expect(saved).toEqual({ backend: 'openvino', host: '10.0.0.5', port: 18765 });
    });

    it('shows why a server cannot be reached', async () => {
        const user = userEvent.setup();
        server.use(
            http.get('/api/system/remote-inference', () =>
                HttpResponse.json({ servers: { torch: { host: '172.18.0.1', port: 1 } } })
            ),
            http.post('/api/system/remote-inference/check', () =>
                HttpResponse.json({ ok: false, url: 'http://172.18.0.1:1', error: 'ConnectError: refused' })
            )
        );
        render(<RemoteServerFields backend='torch' />);

        await waitFor(() => expect(screen.getByRole('textbox', { name: /host/i })).toHaveValue('172.18.0.1'));
        await user.click(screen.getByRole('button', { name: 'Test connection' }));

        expect(await screen.findByText(/Not reachable at http:\/\/172.18.0.1:1: ConnectError: refused/)).toBeVisible();
    });
});

describe('getDefaultInferenceDevice', () => {
    it('never picks the remote server by itself', () => {
        const cpu = { ...remoteDevice, device: 'cpu', type: 'cpu' as const, name: 'CPU' };
        expect(getDefaultInferenceDevice([remoteDevice, cpu], 'torch')?.device).toBe('cpu');
    });
});
