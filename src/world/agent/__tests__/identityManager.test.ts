import { IdentityManager, DEFAULT_IDENTITY_TEMPLATE } from '../identityManager';

function makeManager(): IdentityManager {
  return new IdentityManager('', '');
}

describe('IdentityManager', () => {
  let manager: IdentityManager;

  beforeEach(() => {
    manager = makeManager();
  });

  test('fresh agent gets a default identity record with rich beliefs', async () => {
    const identity = await manager.ensureIdentity('agent_0');
    expect(identity).toBeDefined();
    expect(identity.agent_id).toBe('agent_0');
    expect(identity.name).toBe('agent_0');
    expect(identity.beliefs).toEqual(DEFAULT_IDENTITY_TEMPLATE.beliefs);
    expect(identity.traits).toEqual(DEFAULT_IDENTITY_TEMPLATE.traits);
    expect(identity.edit_count_window).toBe(0);
    expect(identity.window_started_at).toBeNull();
  });

  test('valid single edit succeeds and writes a log entry', async () => {
    const result = await manager.applyIdentityUpdate('agent_0', {
      field: 'name',
      new_value: 'Echo',
      reason: 'Self-naming during first interaction',
    });
    expect(result.ok).toBe(true);
    expect(result.identity).toBeDefined();
    expect(result.identity!.name).toBe('Echo');

    const log = manager.getMockLog();
    expect(log.length).toBe(1);
    expect(log[0].agent_id).toBe('agent_0');
    expect(log[0].field).toBe('name');
    expect(log[0].old_value).toBe('agent_0');
    expect(log[0].new_value).toBe('Echo');
    expect(log[0].reason).toBe('Self-naming during first interaction');
  });

  test('compound update from admin settings succeeds', async () => {
    const result = await manager.applyIdentityUpdate('agent_0', {
      name: 'Nova',
      beliefs: ['I explore physical worlds.'],
      traits: { curiosity: 0.9, persistence: 0.8 },
      reason: 'User configuration in Agent Settings',
    }, true);

    expect(result.ok).toBe(true);
    expect(result.identity!.name).toBe('Nova');
    expect(result.identity!.beliefs).toEqual(['I explore physical worlds.']);
    expect(result.identity!.traits).toEqual({ curiosity: 0.9, persistence: 0.8 });
  });

  test('edit without reason is rejected', async () => {
    const result = await manager.applyIdentityUpdate('agent_0', {
      field: 'name',
      new_value: 'Echo',
      reason: '',
    });
    expect(result.ok).toBe(false);
    expect(result.rejection).toBe('missing_reason');
    expect(manager.getMockLog().length).toBe(0);
  });

  test('autonomous edit inside the 5-min window is rejected with rate_limit', async () => {
    const result1 = await manager.applyIdentityUpdate('agent_0', {
      field: 'name',
      new_value: 'Echo',
      reason: 'First edit',
    });
    expect(result1.ok).toBe(true);

    const result2 = await manager.applyIdentityUpdate('agent_0', {
      field: 'name',
      new_value: 'Echo-2',
      reason: 'Second edit within window',
    });
    expect(result2.ok).toBe(false);
    expect(result2.rejection).toBe('rate_limited');
  });

  test('beliefs append succeeds', async () => {
    const result = await manager.applyIdentityUpdate('agent_0', {
      field: 'beliefs',
      new_value: { op: 'append', entry: 'The world is round' },
      reason: 'Learning about geography',
    });
    expect(result.ok).toBe(true);
    expect(result.identity!.beliefs).toContain('The world is round');
  });
});
