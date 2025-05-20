// Manual mock for child_process to support spyOn and event emitters
const { EventEmitter } = require('events');

function createMockChildProcess() {
    const emitter = new EventEmitter();
    // Provide stdout and stderr streams
    emitter.stdout = new EventEmitter();
    emitter.stderr = new EventEmitter();
    // Forward on and related methods
    emitter.on = emitter.addListener;
    emitter.listeners = emitter.listeners;
    emitter.rawListeners = emitter.rawListeners;
    emitter.listenerCount = emitter.listenerCount;
    emitter.eventNames = emitter.eventNames;
    emitter.kill = jest.fn();
    emitter.send = jest.fn();
    emitter.disconnect = jest.fn();
    emitter.unref = jest.fn();
    emitter.ref = jest.fn();
    emitter.killed = false;
    emitter.connected = true;
    emitter.exitCode = null;
    emitter.signalCode = null;
    emitter.pid = 12345;
    return emitter;
}

module.exports = {
    exec: jest.fn((command, options, callback) => {
        const proc = createMockChildProcess();
        process.nextTick(() => callback && callback(null, '', ''));
        return proc;
    }),
    execSync: jest.fn(),
    spawn: jest.fn((command, args) => {
        const proc = createMockChildProcess();
        return proc;
    }),
};
