import { PythonEnvManager } from '../src/pythonEnvManager';
jest.unmock('child_process');
import * as cp from 'child_process';

jest.mock('child_process');

describe('PythonEnvManager', () => {
    describe('getPythonInterpreter', () => {
        it('should return the system default Python interpreter', async () => {
            jest.spyOn(cp, 'execSync').mockReturnValue('/usr/bin/python3');

            const interpreter = await PythonEnvManager.getPythonInterpreter();
            expect(interpreter).toBe('/usr/bin/python3');
        });

        it('should throw an error if no Python interpreter is found', async () => {
            jest.spyOn(cp, 'execSync').mockImplementation(() => {
                throw new Error('Command not found');
            });

            await expect(PythonEnvManager.getPythonInterpreter()).rejects.toThrow('Command not found');
        });
    });
});
