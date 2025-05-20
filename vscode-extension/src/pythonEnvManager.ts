import * as vscode from 'vscode';
import * as path from 'path';
import * as cp from 'child_process';
import * as fs from 'fs';

/**
 * Utility class for Python environment management in the LazyPredict extension
 */
export class PythonEnvManager {
    /**
     * Find the best available Python interpreter to use
     * Prefers:
     * 1. User-configured interpreter for the extension
     * 2. Active interpreter from VS Code Python extension
     * 3. System default Python
     */
    static async getPythonInterpreter(): Promise<string> {
        // TODO: Add extension settings to allow users to configure a specific Python path
        
        // Try to get the active Python interpreter from the Python extension
        try {
            const pythonExtension = vscode.extensions.getExtension('ms-python.python');
            
            if (pythonExtension) {
                // If Python extension is active, use its API to get the selected interpreter
                if (pythonExtension.isActive) {
                    const pythonAPI = pythonExtension.exports;
                    
                    // Use the Python extension API to get the path
                    if (pythonAPI && pythonAPI.environments) {
                        const activeEnv = await pythonAPI.environments.getActiveEnvironmentPath();
                        if (activeEnv && activeEnv.path) {
                            return activeEnv.path;
                        }
                    }
                } else {
                    // Activate the Python extension if it's not active
                    await pythonExtension.activate();
                    // Then retry getting the interpreter
                    return this.getPythonInterpreter();
                }
            }
        } catch (error) {
            // If there's an error accessing the Python extension, fall back to system Python
            console.log(`Error getting Python path from extension: ${error}`);
        }
        
        // Detect system Python interpreter via execSync
        try {
            const result = cp.execSync('which python3').toString().trim();
            if (result) {
                return result;
            }
            // If no path returned, treat as error
            throw new Error('Python not found');
        } catch (error) {
            // Propagate errors to callers
            throw error;
        }
    }

    /**
     * Check if LazyPredict is installed in the current Python environment
     */
    static async checkLazyPredictInstalled(pythonPath: string = 'python'): Promise<boolean> {
        return new Promise((resolve) => {
            // Run Python with a simple script to check if lazypredict is importable
            const process = cp.spawn(pythonPath, [
                '-c', 'try: import lazypredict; print("success"); except ImportError: print("failed")'
            ]);
            
            let output = '';
            
            process.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            process.on('close', () => {
                resolve(output.trim() === 'success');
            });
            
            process.on('error', () => {
                resolve(false);
            });
        });
    }

    /**
     * Install LazyPredict if it's not already installed
     */
    static async installLazyPredict(pythonPath: string = 'python'): Promise<boolean> {
        return new Promise((resolve, reject) => {
            // Show installation progress
            const installProgress = vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Installing LazyPredict",
                cancellable: true
            }, async (progress, token) => {
                progress.report({ message: "Starting installation..." });
                
                // Install using pip
                const process = cp.spawn(pythonPath, [
                    '-m', 'pip', 'install', 'lazypredict'
                ]);
                
                let stdout = '';
                let stderr = '';
                
                process.stdout.on('data', (data) => {
                    stdout += data.toString();
                    progress.report({ message: "Installing... " });
                });
                
                process.stderr.on('data', (data) => {
                    stderr += data.toString();
                });
                
                process.on('close', (code) => {
                    if (code !== 0) {
                        reject(new Error(`Failed to install LazyPredict: ${stderr}`));
                        return;
                    }
                    
                    progress.report({ message: "LazyPredict installed successfully!" });
                    resolve(true);
                });
                
                process.on('error', (error) => {
                    reject(new Error(`Failed to run pip: ${error.message}`));
                });
                
                token.onCancellationRequested(() => {
                    process.kill();
                    reject(new Error('Installation was cancelled'));
                });
            });
            
            return installProgress;
        });
    }

    /**
     * Check if a Python package is installed
     * @param packageName The name of the package to check
     * @param pythonPath Path to the Python interpreter
     * @returns Promise resolving to a boolean indicating if the package is installed
     */
    static async isPackageInstalled(packageName: string, pythonPath: string = 'python'): Promise<boolean> {
        return new Promise((resolve) => {
            const process = cp.spawn(pythonPath, [
                '-c', `try: import ${packageName}; print("success"); except ImportError: print("failed")`
            ]);
            
            let output = '';
            
            process.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            process.on('close', () => {
                resolve(output.trim() === 'success');
            });
            
            process.on('error', () => {
                resolve(false);
            });
        });
    }
    
    /**
     * Install required packages for LazyPredict
     */
    static async installRequiredPackages(pythonPath: string = 'python'): Promise<boolean> {
        // These are packages that LazyPredict depends on
        const requiredPackages = ['pandas', 'numpy', 'scikit-learn', 'lazypredict'];
        const missingPackages: string[] = [];
        
        // Check which packages are missing
        for (const pkg of requiredPackages) {
            const isInstalled = await this.isPackageInstalled(pkg, pythonPath);
            if (!isInstalled) {
                missingPackages.push(pkg);
            }
        }
        
        if (missingPackages.length === 0) {
            return true; // All packages are installed
        }
        
        // Ask user if they want to install missing packages
        const installChoice = await vscode.window.showInformationMessage(
            `The following packages are required: ${missingPackages.join(', ')}. Would you like to install them?`,
            'Yes', 'No'
        );
        
        if (installChoice !== 'Yes') {
            return false;
        }
        
        // Install missing packages
        return new Promise((resolve, reject) => {
            const installProgress = vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Installing required packages",
                cancellable: true
            }, async (progress, token) => {
                progress.report({ message: "Starting installation..." });
                
                // Install using pip
                const process = cp.spawn(pythonPath, [
                    '-m', 'pip', 'install', ...missingPackages
                ]);
                
                let stdout = '';
                let stderr = '';
                
                process.stdout.on('data', (data) => {
                    stdout += data.toString();
                    progress.report({ message: "Installing... " });
                });
                
                process.stderr.on('data', (data) => {
                    stderr += data.toString();
                });
                
                process.on('close', (code) => {
                    if (code !== 0) {
                        reject(new Error(`Failed to install packages: ${stderr}`));
                        return;
                    }
                    
                    progress.report({ message: "Packages installed successfully!" });
                    resolve(true);
                });
                
                process.on('error', (error) => {
                    reject(new Error(`Failed to run pip: ${error.message}`));
                });
                
                token.onCancellationRequested(() => {
                    process.kill();
                    reject(new Error('Installation was cancelled'));
                });
            });
            
            return installProgress;
        });
    }

    /**
     * Ensure LazyPredict is available, installing it if necessary
     */
    static async ensureLazyPredictAvailable(): Promise<string> {
        // Get Python interpreter
        const pythonPath = await this.getPythonInterpreter();
        
        // Check if LazyPredict is installed
        const isInstalled = await this.checkLazyPredictInstalled(pythonPath);
        
        if (!isInstalled) {
            // Ask user if they want to install LazyPredict
            const installChoice = await vscode.window.showInformationMessage(
                'LazyPredict is not installed in the current Python environment. Would you like to install it?',
                'Yes', 'No'
            );
            
            if (installChoice === 'Yes') {
                try {
                    await this.installLazyPredict(pythonPath);
                    vscode.window.showInformationMessage('LazyPredict has been installed successfully!');
                } catch (error: any) {
                    vscode.window.showErrorMessage(`Failed to install LazyPredict: ${error.message}`);
                    throw new Error(`LazyPredict is not available: ${error.message}`);
                }
            } else {
                throw new Error('LazyPredict is required but not installed.');
            }
        }
        
        return pythonPath;
    }
}
