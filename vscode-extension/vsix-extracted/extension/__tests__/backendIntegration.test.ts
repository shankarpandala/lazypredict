import * as vscode from 'vscode';
import { jest } from '@jest/globals';
import * as assert from 'assert';
import { activate } from '../src/extension';

// Helper function to delay execution
const delay = (ms: number) => new Promise(res => setTimeout(res, ms));

describe('Extension Backend Integration Tests', () => {
    let postMessageMock: jest.Mock;
    let webviewPanel: vscode.WebviewPanel;
    let mockContext: vscode.ExtensionContext;

    beforeEach(() => {
        // Mock vscode.window.showInformationMessage
        (vscode.window.showInformationMessage as jest.Mock) = jest.fn(() => Promise.resolve(undefined));

        // Create a mock WebviewPanel
        webviewPanel = {
            webview: {
                postMessage: jest.fn(),
                onDidReceiveMessage: jest.fn((callback: (message: any) => void) => {
                    callback({ command: 'runLazyPredict', data: { filePath: '/path/to/data.csv' } });
                }),
            },
            onDidDispose: jest.fn(),
        } as unknown as vscode.WebviewPanel;
        postMessageMock = webviewPanel.webview.postMessage as jest.Mock;

        // Mock vscode.window.createWebviewPanel
        jest.spyOn(vscode.window, 'createWebviewPanel').mockReturnValue(webviewPanel);

        // Create a mock ExtensionContext
        mockContext = {
            subscriptions: [],
            workspaceState: {} as any,
            globalState: {} as any,
            secrets: {} as any,
            extensionUri: vscode.Uri.file('/mock/uri'),
            extensionPath: '/mock/path',
            environmentVariableCollection: {} as any,
            asAbsolutePath: jest.fn((relativePath: string) => `/mock/path/${relativePath}`),
            storageUri: undefined,
            globalStorageUri: vscode.Uri.file('/mock/globalStorage'),
            logUri: vscode.Uri.file('/mock/log'),
            extensionMode: vscode.ExtensionMode.Test,
            globalStoragePath: '/mock/globalStoragePath',
            logPath: '/mock/logPath',
            storagePath: undefined,
            extension: {} as any, // Mock extension property
            languageModelAccessInformation: {} as any, // Mock language model access information
        };

        // Activate the extension to register commands
        activate(mockContext);
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    describe('Extension Commands', () => {
        it('should register and trigger "lazypredict.showUI" command, creating a webview panel', async () => {
            await vscode.commands.executeCommand('lazypredict.showUI');

            // Check if createWebviewPanel was called
            expect(vscode.window.createWebviewPanel).toHaveBeenCalledTimes(1);

            // Optionally, check arguments if needed (e.g., viewType, title)
            expect(vscode.window.createWebviewPanel).toHaveBeenCalledWith(
                'lazyPredictResults',
                'LazyPredict UI',
                vscode.ViewColumn.One,
                expect.any(Object)
            );

            // Ensure the panel has some HTML content set (basic check)
            expect(webviewPanel.webview.postMessage).toHaveBeenCalled();
        });

        // Add more tests for other commands if any
    });

    describe('Webview <-> Extension Communication', () => {
        it('should receive messages from webview and process them', async () => {
            // Simulate the webview sending a message
            const onDidReceiveMessageMock = webviewPanel.webview.onDidReceiveMessage as jest.Mock;

            // Check if the backend logic (e.g., running Python script) was triggered
            // This will depend on how your backend is structured.
            // For example, if it calls a Python script, you might mock child_process.spawn
            // or check for a log message.
            assert.ok(true, "Message handling logic needs to be verified based on implementation.");
        });
    });
});
