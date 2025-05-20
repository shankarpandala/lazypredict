import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as cp from 'child_process';
import { WebviewToExtensionMessage, ExtensionToWebviewMessage } from './types';
import { PythonEnvManager } from './pythonEnvManager';
import { FileUtils } from './fileUtils';

// Function to run the LazyPredict Python script
export async function runLazyPredictScript(
    filePath: string, 
    targetColumn: string, 
    problemType: 'classification' | 'regression' = 'classification',
    testSize: number = 0.2, 
    randomState: number = 42,
    customModels?: string[]): Promise<any> {
    
    return new Promise(async (resolve, reject) => {
        try {
            // Ensure LazyPredict is available and get the Python interpreter path
            const pythonCommand = await PythonEnvManager.ensureLazyPredictAvailable();
            
            // Get the path to the Python script
            const scriptPath = path.join(__dirname, '..', 'src', 'python', 'run_lazypredict.py');
            
            // Build the command arguments
            const args = [
                scriptPath,
                filePath,
                targetColumn,
                problemType,
                testSize.toString(),
                randomState.toString()
            ];
            
            // Add custom models if provided
            if (customModels && customModels.length > 0) {
                args.push(customModels.join(','));
            }
            
            // Execute the Python script
            const process = cp.spawn(pythonCommand, args);
            
            let stdout = '';
            let stderr = '';
            
            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            process.on('close', (code) => {
                if (code !== 0) {
                    reject(new Error(`Python script exited with code ${code}. Error: ${stderr}`));
                    return;
                }
                
                try {
                    // Parse the JSON output from the script
                    const results = JSON.parse(stdout);
                    
                    // Check if there was an error in the Python script
                    if (results.error) {
                        reject(new Error(results.error));
                        return;
                    }
                    
                    resolve(results);
                } catch (error) {
                    reject(new Error(`Failed to parse Python script output: ${error}. Output was: ${stdout}`));
                }
            });
            
            process.on('error', (error) => {
                reject(new Error(`Failed to execute Python script: ${error.message}`));
            });
        } catch (error) {
            reject(error);
        }
    });
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
    // Use the console to output diagnostic information (console.log)
    // and errors (console.error)
    console.log('Congratulations, your extension "lazypredict-ui" is now active!');

    // Register command to open dataset file
    context.subscriptions.push(
        vscode.commands.registerCommand('lazypredict.openDatasetFile', async () => {
            try {
                // Show file picker dialog to select CSV or Excel file
                const fileUris = await vscode.window.showOpenDialog({
                    canSelectFiles: true,
                    canSelectFolders: false,
                    canSelectMany: false,
                    filters: {
                        'Data files': ['csv', 'xlsx', 'xls']
                    },
                    title: 'Select Dataset File'
                });

                if (fileUris && fileUris.length > 0) {
                    return fileUris[0].fsPath;
                }
                return undefined;
            } catch (error: any) {
                vscode.window.showErrorMessage(`Error opening dataset: ${error.message}`);
                return undefined;
            }
        })
    );

    // Register the main command to show the LazyPredict UI
    let disposable = vscode.commands.registerCommand('lazypredict.showUI', () => {
        // Create and show a new webview panel
        const panel = vscode.window.createWebviewPanel(
            'lazyPredictResults', // Identifies the type of the webview. Used internally
            'LazyPredict UI', // Updated title
            vscode.ViewColumn.One, // Editor column to show the new webview panel in.
            {
                enableScripts: true,
                retainContextWhenHidden: true, // Keep state when webview is not visible
                localResourceRoots: [
                    vscode.Uri.joinPath(context.extensionUri, 'dist'), // For bundled React app
                    vscode.Uri.joinPath(context.extensionUri, 'media') // For other static assets
                ]
            }
        );

        // Get path to React build output
        const buildPath = vscode.Uri.joinPath(context.extensionUri, 'dist', 'index.html');
        let htmlContent = fs.readFileSync(buildPath.fsPath, 'utf8');

        // Replace placeholders for script and style sources with webview URIs
        htmlContent = htmlContent.replace(
            /(href|src)="\/static\//g,
            `$1="${panel.webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'dist', 'static'))}/`
        );
         // Also replace the main bundle paths if they are at root
        htmlContent = htmlContent.replace(
            /(href|src)="\/(main\..*\.(js|css))"/g,
            `$1="${panel.webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'dist'))}/$2"`
        );

        panel.webview.html = htmlContent;

        // Listen for messages from the webview
        panel.webview.onDidReceiveMessage(
            async (message: WebviewToExtensionMessage) => {
                switch (message.command) {
                    case 'runLazyPredict':
                        vscode.window.showInformationMessage(`Running LazyPredict with: ${JSON.stringify(message.data)}`);
                        
                        // Send loading message to webview
                        panel.webview.postMessage({ 
                            command: 'showLoading', 
                            isLoading: true 
                        } as ExtensionToWebviewMessage);
                        
                        try {
                            // Run the LazyPredict Python script with the provided data
                            const { filePath, problemType, customModels } = message.data;
                            
                            // Get target column using the FileUtils helper
                            const targetColumn = await FileUtils.promptForTargetColumn(filePath);
                            
                            // Run the analysis
                            const results = await runLazyPredictScript(
                                filePath, 
                                targetColumn, 
                                problemType,
                                0.2, // default test size
                                42,  // default random state
                                customModels
                            );
                            
                            // Send the results back to the webview
                            panel.webview.postMessage({ 
                                command: 'analysisResult', 
                                data: results 
                            } as ExtensionToWebviewMessage);
                            
                            // Show a success message
                            vscode.window.showInformationMessage(
                                `LazyPredict analysis complete. Best model: ${results.bestModel}`
                            );
                        } catch (error: any) {
                            // Send error to webview
                            panel.webview.postMessage({ 
                                command: 'analysisError', 
                                error: error.message 
                            } as ExtensionToWebviewMessage);
                            
                            // Also show error in VS Code UI
                            vscode.window.showErrorMessage(`LazyPredict Error: ${error.message}`);
                        } finally {
                            // Hide loading indicator
                            panel.webview.postMessage({ 
                                command: 'showLoading', 
                                isLoading: false 
                            } as ExtensionToWebviewMessage);
                        }
                        return;
                    case 'getInitialData':
                        // Example: send some initial data or settings if needed
                        panel.webview.postMessage({ command: 'initialData', data: { lastRunSettings: {} } } as ExtensionToWebviewMessage);
                        return;
                    case 'showInfo':
                        vscode.window.showInformationMessage(message.text);
                        return;
                }
            },
            undefined,
            context.subscriptions
        );

        // Handle panel disposal
        panel.onDidDispose(
            () => {
                // Clean up resources when the panel is closed
                // This can be used to save state, cancel running tasks, etc.
                console.log('LazyPredict UI panel closed');
            },
            null,
            context.subscriptions
        );
    });

    // Register command to get available Python environment info
    context.subscriptions.push(
        vscode.commands.registerCommand('lazypredict.getPythonInfo', async () => {
            try {
                const pythonPath = await PythonEnvManager.getPythonInterpreter();
                const isLazyPredictInstalled = await PythonEnvManager.checkLazyPredictInstalled(pythonPath);
                
                return {
                    pythonPath,
                    isLazyPredictInstalled
                };
            } catch (error: any) {
                vscode.window.showErrorMessage(`Error getting Python info: ${error.message}`);
                return { pythonPath: undefined, isLazyPredictInstalled: false };
            }
        })
    );

    // Register command to view analysis results
    context.subscriptions.push(
        vscode.commands.registerCommand('lazypredict.viewResults', async (results: any) => {
            // Open a new editor with the results as markdown
            if (!results) {
                vscode.window.showErrorMessage('No results to display');
                return;
            }

            try {
                const doc = await vscode.workspace.openTextDocument({
                    content: generateMarkdownResults(results),
                    language: 'markdown'
                });
                
                await vscode.window.showTextDocument(doc);
            } catch (error: any) {
                vscode.window.showErrorMessage(`Error displaying results: ${error.message}`);
            }
        })
    );

    // Add all disposables to context subscriptions
    context.subscriptions.push(disposable);
}

/**
 * Generate markdown representation of LazyPredict results
 */
function generateMarkdownResults(results: any): string {
    if (!results || results.error) {
        return `# LazyPredict Analysis Error\n\n${results?.error || 'Unknown error'}`;
    }

    let markdown = `# LazyPredict Analysis Results\n\n`;
    
    markdown += `## Problem Type: ${results.problemType}\n\n`;
    markdown += `## Best Model: ${results.bestModel}\n\n`;
    
    markdown += `## Model Performance\n\n`;
    
    // Create a markdown table of model results
    if (results.models && results.models.length > 0) {
        const headers = Object.keys(results.models[0]);
        
        markdown += `| ${headers.join(' | ')} |\n`;
        markdown += `| ${headers.map(() => '---').join(' | ')} |\n`;
        
        for (const model of results.models) {
            markdown += `| ${headers.map(h => model[h]).join(' | ')} |\n`;
        }
    }
    
    if (results.logs) {
        markdown += `\n## Logs\n\n\`\`\`\n${results.logs}\n\`\`\`\n`;
    }
    
    return markdown;
}

// This method is called when your extension is deactivated
export function deactivate() {
    // Clean up any resources that need to be released when the extension is deactivated
}
