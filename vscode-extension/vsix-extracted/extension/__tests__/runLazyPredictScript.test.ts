import { runLazyPredictScript } from '../src/extension';
import * as cp from 'child_process';

// Use the manual child_process mock

jest.mock('../src/pythonEnvManager', () => ({
    PythonEnvManager: {
        ensureLazyPredictAvailable: jest.fn(() => Promise.resolve('python')),
    },
}));

describe('runLazyPredictScript', () => {
    it('should execute the LazyPredict script and return results', async () => {
        jest.spyOn(cp, 'exec').mockImplementation((command, options, callback) => {
            const mockChildProcess = {
                stdin: null,
                stdout: null,
                stderr: null,
                stdio: [null, null, null],
                on: jest.fn(),
                listeners: jest.fn(),
                rawListeners: jest.fn(),
                listenerCount: jest.fn(),
                eventNames: jest.fn(),
                kill: jest.fn(),
                send: jest.fn(),
                disconnect: jest.fn(),
                unref: jest.fn(),
                ref: jest.fn(),
                killed: false,
                connected: true,
                exitCode: null,
                signalCode: null,
                pid: 12345,
            };
            process.nextTick(() => callback?.(null, 'Mocked LazyPredict Results', ''));
            return mockChildProcess as unknown as cp.ChildProcess;
        });

        const results = await runLazyPredictScript('/path/to/data.csv', 'target', 'classification');
        expect(results).toBe('Mocked LazyPredict Results');
    });

    it('should throw an error if the script execution fails', async () => {
        jest.spyOn(cp, 'exec').mockImplementation((command, options, callback) => {
            const mockChildProcess = {
                stdin: null,
                stdout: null,
                stderr: null,
                stdio: [null, null, null],
                on: jest.fn(),
                listeners: jest.fn(),
                rawListeners: jest.fn(),
                listenerCount: jest.fn(),
                eventNames: jest.fn(),
                kill: jest.fn(),
                send: jest.fn(),
                disconnect: jest.fn(),
                unref: jest.fn(),
                ref: jest.fn(),
                killed: false,
                connected: true,
                exitCode: null,
                signalCode: null,
                pid: 12345,
            };
            process.nextTick(() => callback?.(new Error('Execution failed'), '', 'Error output'));
            return mockChildProcess as unknown as cp.ChildProcess;
        });

        await expect(runLazyPredictScript('/path/to/data.csv', 'target', 'classification')).rejects.toThrow('Execution failed');
    });
});
