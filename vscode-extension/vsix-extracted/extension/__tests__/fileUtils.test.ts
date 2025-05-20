import { FileUtils } from '../src/fileUtils';
import * as fs from 'fs';
import * as path from 'path';
import * as child_process from 'child_process';

// Use real fs for readFileSync in tests
jest.unmock('fs');
jest.mock('child_process');

// Mock PythonEnvManager to avoid actual Python calls
jest.mock('../src/pythonEnvManager', () => ({
    PythonEnvManager: {
        ensureLazyPredictAvailable: jest.fn(() => Promise.resolve('python')),
    },
}));

describe('FileUtils', () => {
    describe('getDatasetColumns', () => {
        it('should return column headers for a valid CSV file', async () => {
            // Mock file content
            const mockCsvContent = 'col1,col2,col3\n1,2,3\n4,5,6';
            jest.spyOn(fs, 'readFileSync').mockReturnValue(mockCsvContent);

            const columns = await FileUtils.getDatasetColumns('/path/to/mock.csv');
            expect(columns).toEqual(['col1', 'col2', 'col3']);
        });

        it('should throw an error for an invalid file', async () => {
            jest.spyOn(fs, 'readFileSync').mockImplementation(() => {
                throw new Error('File not found');
            });

            await expect(FileUtils.getDatasetColumns('/path/to/invalid.csv')).rejects.toThrow('File not found');
        });

        it('should return column headers by executing read_headers.py for a valid file', async () => {
            const mockExec = jest.spyOn(child_process, 'exec').mockImplementation((command, options, callback) => {
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
                process.nextTick(() => {
                    if (command.includes('read_headers.py')) {
                        callback?.(null, 'col1,col2,col3', '');
                    } else {
                        callback?.(new Error('File not found'), '', '');
                    }
                });
                return mockChildProcess as unknown as child_process.ChildProcess;
            });

            const columns = await FileUtils.getDatasetColumns('/path/to/mock.csv');
            expect(columns).toEqual(['col1', 'col2', 'col3']);
            expect(mockExec).toHaveBeenCalledWith(
                expect.stringContaining('read_headers.py'),
                expect.anything(),
                expect.anything()
            );
        });
    });
});
