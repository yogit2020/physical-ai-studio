import { useEffect, useState } from 'react';

import { Button, Flex, Heading, StatusLight, Text, toast } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSafetyConfig, SchemaSessionSafety } from '../../../api/openapi-spec';
import { ERROR_TOAST, useRuntimeSession } from '../runtime-session-provider';

type SafetyAction = 'monitor' | 'acknowledge' | 'stop';

const HEALTH_LIGHT: Record<SchemaSessionSafety['monitor']['health'], 'positive' | 'notice' | 'negative' | 'neutral'> = {
    ok: 'positive',
    degraded: 'notice',
    down: 'negative',
    off: 'neutral',
};

const latchText = (safety: SchemaSessionSafety): string => {
    const cause = safety.latched_by;
    const by = cause ? ` by ${cause.source}${cause.reason ? `: ${cause.reason}` : ''}` : '';
    if (safety.state === 'homing') {
        return `Moving home${by}`;
    }
    return safety.state === 'latched' ? `Stopped${by}` : 'Clear';
};

/**
 * The config to send to turn the monitor on or off. Only the monitor changes: ``home_pose`` and
 * ``stop_controls`` are left out so the backend keeps its own values (a home pose captured meanwhile
 * survives); the other top-level fields are sent back as read. The monitor is sent whole, as read,
 * with only ``enabled`` flipped, so its saved endpoint, model, cameras and prompt are kept.
 */
export const withMonitorEnabled = (config: SchemaSafetyConfig, enabled: boolean): SchemaSafetyConfig => ({
    on_stop: config.on_stop,
    home_duration_s: config.home_duration_s,
    max_joint_speed: config.max_joint_speed,
    monitor: config.monitor && { ...config.monitor, enabled },
});

/**
 * The session's safety latch and vision monitor: what stopped the follower, whether the monitor
 * watches, and the operator's buttons to turn it on or off, acknowledge a stop, or stop now.
 */
export const SafetyPanel = () => {
    const { state, send } = useRuntimeSession();
    const [pending, setPending] = useState<SafetyAction | null>(null);
    const session_id = state?.session_id;
    const safety = state?.safety;

    // The live latch and monitor health come with every session state; the saved config only from here.
    const { data: status, refetch } = $api.useQuery(
        'get',
        '/api/runtime/sessions/{session_id}/safety',
        { params: { path: { session_id: session_id ?? '' } } },
        { enabled: session_id !== undefined }
    );

    // The session sends a new state after each safety event, monitor health change and load.
    const safetyKey = JSON.stringify([safety ?? null, state?.environment?.robot_id ?? null]);
    useEffect(() => {
        if (session_id !== undefined) {
            void refetch();
        }
    }, [safetyKey, session_id, refetch]);

    if (state == null || safety === undefined) {
        return null;
    }

    const config = status?.config;
    const monitorEnabled = config?.monitor?.enabled ?? false;
    const hasEnvironment = state.environment != null;
    const isLatched = safety.latched_by != null;

    const run = async (action: SafetyAction, command: Parameters<typeof send>[0], failure: string) => {
        setPending(action);
        try {
            const ack = await send(command);
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? failure, ERROR_TOAST);
            }
        } finally {
            setPending(null);
            void refetch();
        }
    };

    const toggleMonitor = () => {
        if (config === undefined) {
            return;
        }
        void run(
            'monitor',
            { command: 'set_safety', config: withMonitorEnabled(config, !monitorEnabled) },
            'Could not change the safety monitor.'
        );
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Heading level={4} margin={0}>
                Safety
            </Heading>
            <StatusLight variant={isLatched || safety.state !== 'clear' ? 'negative' : 'positive'}>
                {latchText(safety)}
            </StatusLight>
            <StatusLight variant={monitorEnabled ? HEALTH_LIGHT[safety.monitor.health] : 'neutral'}>
                {`Monitor ${monitorEnabled ? 'on' : 'off'}, ${safety.monitor.health}${
                    safety.monitor.guarding ? ', guarding' : ''
                }`}
            </StatusLight>
            {safety.monitor.error && <Text>{safety.monitor.error}</Text>}
            {monitorEnabled && safety.monitor.health === 'off' && (
                <Text>The monitor is on but not checking: no endpoint is configured or reachable.</Text>
            )}
            <Button
                variant='secondary'
                isPending={pending === 'monitor'}
                isDisabled={config === undefined || pending !== null}
                onPress={toggleMonitor}
            >
                {monitorEnabled ? 'Disable safety monitor' : 'Enable safety monitor'}
            </Button>
            <Button
                variant='secondary'
                isPending={pending === 'acknowledge'}
                isDisabled={!isLatched || pending !== null}
                onPress={() =>
                    void run(
                        'acknowledge',
                        { command: 'safety_acknowledge', source: 'operator' },
                        'Could not acknowledge the stop.'
                    )
                }
            >
                Acknowledge
            </Button>
            <Button
                variant='negative'
                isPending={pending === 'stop'}
                isDisabled={!hasEnvironment || pending !== null}
                onPress={() =>
                    void run(
                        'stop',
                        {
                            command: 'safety_stop',
                            reason: 'Emergency stop from the UI',
                            source: 'ui',
                            hazard: 'other',
                            hold: false,
                        },
                        'Could not stop.'
                    )
                }
            >
                Emergency stop
            </Button>
        </Flex>
    );
};
