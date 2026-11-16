import { useEffect, useRef, useState } from 'react';

import { Button, Flex, NumberField, Text, TextField } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaRemoteServerCheck } from '../../../api/openapi-spec';

export const REMOTE_DEVICE = 'REMOTE';

/** One line on what the server answered: reachable, how fast, what it runs, and whether it is this model. */
const CheckResult = ({ check }: { check: SchemaRemoteServerCheck }) => {
    if (!check.ok) {
        return (
            <Text UNSAFE_style={{ color: 'var(--spectrum-negative-visual-color)' }}>
                ✗ Not reachable at {check.url}: {check.error}
            </Text>
        );
    }
    const runs = `${check.backend ?? '?'} on ${check.device ?? '?'}, ${check.latency_ms?.toFixed(0) ?? '?'} ms`;
    if (check.same_model === false) {
        return (
            <Text UNSAFE_style={{ color: 'var(--spectrum-notice-visual-color)' }}>
                ⚠ Connected ({runs}), but it runs another model than this one. Loading will be refused.
            </Text>
        );
    }
    return (
        <Text UNSAFE_style={{ color: 'var(--spectrum-positive-visual-color)' }}>
            ✓ Connected ({runs}){check.same_model ? ', same model' : ''}
        </Text>
    );
};

interface RemoteServerFieldsProps {
    backend: 'openvino' | 'torch';
    modelId?: string;
}

/**
 * Host and port of the remote policy server a model on ``backend`` runs on. Saved per backend, so
 * the next load is pre-filled; "Test connection" saves, then asks the server what it runs.
 */
export const RemoteServerFields = ({ backend, modelId }: RemoteServerFieldsProps) => {
    const { data: config } = $api.useQuery('get', '/api/system/remote-inference');
    const saved = config?.servers?.[backend];
    const [host, setHost] = useState(saved?.host ?? '');
    const [port, setPort] = useState<number>(saved?.port ?? Number.NaN);
    const [check, setCheck] = useState<SchemaRemoteServerCheck | undefined>();

    // Each backend has its own server: start over when it changes.
    useEffect(() => {
        setHost('');
        setPort(Number.NaN);
        setCheck(undefined);
    }, [backend]);
    // The saved server arrives after the first render; fill the fields while they are still empty,
    // never over what was typed, and not again after a save refreshes it.
    useEffect(() => {
        if (saved !== undefined) {
            setHost((current) => (current === '' ? saved.host : current));
            setPort((current) => (Number.isNaN(current) ? saved.port : current));
        }
    }, [saved]);

    const save = $api.useMutation('put', '/api/system/remote-inference/{backend}', {
        meta: {
            invalidates: [
                ['get', '/api/system/remote-inference'],
                ['get', '/api/system/devices/inference'],
            ],
        },
    });
    const test = $api.useMutation('post', '/api/system/remote-inference/check');

    const isValid = host.trim().length > 0 && Number.isInteger(port) && port >= 1 && port <= 65535;
    const isChanged = host.trim() !== (saved?.host ?? '') || port !== saved?.port;

    // The port field commits its number when it loses focus, which pressing the button causes, so the
    // press can run before the re-render: read the newest values, not this render's.
    const latest = useRef({ host, port });
    latest.current = { host, port };

    const saveAndTest = async () => {
        const current = { host: latest.current.host.trim(), port: latest.current.port };
        const valid =
            current.host.length > 0 && Number.isInteger(current.port) && current.port >= 1 && current.port <= 65535;
        if (!valid) {
            return;
        }
        setCheck(undefined);
        if (current.host !== saved?.host || current.port !== saved?.port) {
            await save.mutateAsync({ params: { path: { backend } }, body: current });
        }
        setCheck(await test.mutateAsync({ body: { ...current, ...(modelId ? { model_id: modelId, backend } : {}) } }));
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Flex gap='size-100' alignItems='end'>
                <TextField
                    label='Remote server host'
                    width='100%'
                    value={host}
                    onChange={(value) => {
                        latest.current.host = value;
                        setHost(value);
                        setCheck(undefined);
                    }}
                    description='For example 172.18.0.1'
                    validationState={host.includes('://') ? 'invalid' : undefined}
                    errorMessage='Host or IP only, without http://'
                    isRequired
                />
                <NumberField
                    label='Port'
                    width='size-1600'
                    value={port}
                    onChange={(value) => {
                        latest.current.port = value;
                        setPort(value);
                        setCheck(undefined);
                    }}
                    minValue={1}
                    maxValue={65535}
                    formatOptions={{ useGrouping: false }}
                    hideStepper
                    isRequired
                />
            </Flex>
            <Flex gap='size-150' alignItems='center' wrap>
                <Button
                    variant='secondary'
                    onPress={() => void saveAndTest()}
                    isPending={save.isPending || test.isPending}
                    isDisabled={!isValid}
                >
                    {isChanged ? 'Save and test connection' : 'Test connection'}
                </Button>
                {check !== undefined && <CheckResult check={check} />}
                {check === undefined && isChanged && isValid && (
                    <Text>Not saved yet: loading uses the saved server.</Text>
                )}
            </Flex>
        </Flex>
    );
};
