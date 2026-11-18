import { toast } from '@geti-ui/ui';
import { afterEach, beforeEach, describe, expect, it, rs } from '@rstest/core';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaAckMessage, SchemaSessionState } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedMonitorConfig, getMockedSafetyStatus } from '../../../test-utils/mocks/mock-safety-status';
import { render } from '../../../test-utils/render';
import { SafetyPanel } from './safety-panel';

const SAFETY_PATH = '/api/runtime/sessions/{session_id}/safety';

const session: { state: SchemaSessionState; send: ReturnType<typeof rs.fn> } = {
    state: {
        session_id: 'session-1',
        safety: { state: 'clear', home_pose_set: true, monitor: { health: 'off', guarding: false } },
    },
    send: rs.fn(),
};

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: { timeout: 10_000 },
    useRuntimeSession: () => session,
}));

const ok = (): Promise<SchemaAckMessage> => Promise.resolve({ type: 'ack', request_id: 'r', ok: true });

beforeEach(() => {
    session.send = rs.fn(ok);
    session.state = {
        session_id: 'session-1',
        environment: {
            environment_id: 'env-1',
            name: 'Bench',
            robot_id: 'robot-1',
            leader_id: null,
            joint_names: [],
            cameras: [],
        },
        safety: { state: 'clear', home_pose_set: true, monitor: { health: 'off', guarding: false } },
    };
});

afterEach(() => {
    rs.restoreAllMocks();
});

describe('SafetyPanel', () => {
    it('enables the monitor with its saved endpoint, cameras and prompt', async () => {
        server.use(http.get(SAFETY_PATH, () => HttpResponse.json(getMockedSafetyStatus())));

        render(<SafetyPanel />);

        expect(screen.getByText('Clear')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Acknowledge' })).toBeDisabled();
        const enable = await screen.findByRole('button', { name: 'Enable safety monitor' });
        await waitFor(() => expect(enable).toBeEnabled());

        fireEvent.click(enable);

        await waitFor(() => expect(session.send).toHaveBeenCalledTimes(1));
        expect(session.send).toHaveBeenCalledWith({
            command: 'set_safety',
            config: {
                on_stop: 'home',
                home_duration_s: 4,
                max_joint_speed: 45,
                monitor: getMockedMonitorConfig({ enabled: true }),
            },
        });
    });

    it('disables a monitor that is on', async () => {
        server.use(
            http.get(SAFETY_PATH, () =>
                HttpResponse.json(
                    getMockedSafetyStatus({
                        config: {
                            ...getMockedSafetyStatus().config,
                            monitor: getMockedMonitorConfig({ enabled: true }),
                        },
                    })
                )
            )
        );

        render(<SafetyPanel />);
        fireEvent.click(await screen.findByRole('button', { name: 'Disable safety monitor' }));

        await waitFor(() =>
            expect(session.send).toHaveBeenCalledWith(
                expect.objectContaining({ config: expect.objectContaining({ monitor: getMockedMonitorConfig() }) })
            )
        );
    });

    it('shows why the follower stopped and acknowledges the stop', async () => {
        server.use(http.get(SAFETY_PATH, () => HttpResponse.json(getMockedSafetyStatus({ state: 'latched' }))));
        session.state = {
            ...session.state,
            safety: {
                state: 'latched',
                home_pose_set: true,
                monitor: { health: 'ok', guarding: false },
                latched_by: {
                    id: 'event-1',
                    at: '2026-11-18T10:00:00Z',
                    kind: 'stop',
                    source: 'monitor',
                    reason: 'hand on the mat',
                },
            },
        };

        render(<SafetyPanel />);

        expect(screen.getByText('Stopped by monitor: hand on the mat')).toBeInTheDocument();
        const acknowledge = screen.getByRole('button', { name: 'Acknowledge' });
        expect(acknowledge).toBeEnabled();

        fireEvent.click(acknowledge);

        await waitFor(() =>
            expect(session.send).toHaveBeenCalledWith({ command: 'safety_acknowledge', source: 'operator' })
        );
    });

    it('shows the error when the session refuses a command', async () => {
        const notify = rs.spyOn(toast, 'negative').mockReturnValue(rs.fn());
        server.use(http.get(SAFETY_PATH, () => HttpResponse.json(getMockedSafetyStatus())));
        session.send = rs.fn(() =>
            Promise.resolve({
                type: 'ack',
                request_id: 'r',
                ok: false,
                error: { message: 'Load an environment first.', error_code: 'command_failed' },
            })
        );

        render(<SafetyPanel />);
        fireEvent.click(screen.getByRole('button', { name: 'Emergency stop' }));

        await waitFor(() => expect(notify).toHaveBeenCalledWith('Load an environment first.', { timeout: 10_000 }));
        expect(session.send).toHaveBeenCalledWith(
            expect.objectContaining({ command: 'safety_stop', source: 'ui', hold: false })
        );
    });
});
