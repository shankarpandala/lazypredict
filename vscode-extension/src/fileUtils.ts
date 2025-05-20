import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as cp from 'child_process';
import { PythonEnvManager } from './pythonEnvManager';

/**
 * Utility functions for file operations related to LazyPredict
 */
export class FileUtils {
    /**
     * Get the column headers from a CSV or Excel file
     * @param filePath Path to the file
     * @returns Promise resolving to an array of column headers
     */
    static async getDatasetColumns(filePath: string): Promise<string[]> {
        return new Promise(async (resolve, reject) => {
            try {
                // Ensure we have Python with the required libraries
                const pythonCommand = await PythonEnvManager.ensureLazyPredictAvailable();
                
                // Use a simple Python script to read headers
                const pythonScript = `
import pandas as pd
import sys
import json

try:
    if sys.argv[1].lower().endswith('.csv'):
        df = pd.read_csv(sys.argv[1], nrows=1)
    elif sys.argv[1].lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(sys.argv[1], nrows=1)
    else:
        sys.stderr.write('Unsupported file format')
        sys.exit(1)
    
    headers = df.columns.tolist()
    print(json.dumps(headers))
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
`;
                
                // Write the Python script to a temporary file
                const tmpDir = path.join(__dirname, 'tmp');
                if (!fs.existsSync(tmpDir)) {
                    fs.mkdirSync(tmpDir, { recursive: true });
                }
                
                const scriptPath = path.join(tmpDir, 'read_headers.py');
                fs.writeFileSync(scriptPath, pythonScript);
                
                // Run the script
                const process = cp.spawn(pythonCommand, [scriptPath, filePath]);
                
                let stdout = '';
                let stderr = '';
                
                process.stdout.on('data', (data) => {
                    stdout += data.toString();
                });
                
                process.stderr.on('data', (data) => {
                    stderr += data.toString();
                });
                
                process.on('close', (code) => {
                    // Clean up the temporary script
                    try {
                        fs.unlinkSync(scriptPath);
                    } catch (e) {
                        console.error('Failed to delete temporary script:', e);
                    }
                    
                    if (code !== 0) {
                        reject(new Error(`Failed to read dataset headers: ${stderr}`));
                        return;
                    }
                    
                    try {
                        const headers = JSON.parse(stdout.trim());
                        resolve(headers);
                    } catch (e) {
                        reject(new Error(`Failed to parse headers: ${e}. Output: ${stdout}`));
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

    /**
     * Prompt the user to select the target column from a dataset
     * @param filePath Path to the dataset file
     * @returns Promise resolving to the selected column name
     */
    static async promptForTargetColumn(filePath: string): Promise<string> {
        try {
            // Get columns from the dataset
            const columns = await this.getDatasetColumns(filePath);
            
            if (!columns || columns.length === 0) {
                throw new Error('No columns found in the dataset');
            }
            
            // Show quickpick to select target column
            const selectedColumn = await vscode.window.showQuickPick(columns, {
                placeHolder: 'Select the target column for prediction',
                canPickMany: false
            });
            
            if (!selectedColumn) {
                throw new Error('Target column selection was cancelled');
            }
            
            return selectedColumn;
        } catch (error: any) {
            // If we can't read the columns automatically, fall back to manual input
            const manualColumn = await vscode.window.showInputBox({
                prompt: 'Enter the name of the target column',
                placeHolder: 'e.g., target, price, class, etc.',
                ignoreFocusOut: true
            });
            
            if (!manualColumn) {
                throw new Error('Target column is required to run analysis');
            }
            
            return manualColumn;
        }
    }
}
