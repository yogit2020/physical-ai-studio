import { SchemaMonitorConfig, SchemaSafetyStatus } from '../../api/openapi-spec';

export const getMockedMonitorConfig = (overrides: Partial<SchemaMonitorConfig> = {}): SchemaMonitorConfig => ({
    enabled: false,
    endpoint: 'http://monitor:18766/v1',
    model: 'qwen-safety',
    cameras: ['arm-camera', 'topview'],
    check_hz: 2,
    image_max_side: 448,
    timeout_s: 3,
    min_confidence: 0.6,
    unsafe_consecutive: 1,
    fail_safe_after_s: 2,
    max_frame_age_s: 1,
    prompt: 'Judge only the white mat.',
    ...overrides,
});

export const getMockedSafetyStatus = (overrides: Partial<SchemaSafetyStatus> = {}): SchemaSafetyStatus => ({
    session_id: 'session-1',
    state: 'clear',
    latched_by: null,
    home_pose_set: true,
    monitor: { health: 'off', guarding: false },
    config: {
        monitor: getMockedMonitorConfig(),
        on_stop: 'home',
        home_pose: [0, 0, 0, 0, 0, 0],
        home_duration_s: 4,
        max_joint_speed: 45,
        stop_controls: ['model'],
    },
    counters: {},
    ...overrides,
});
