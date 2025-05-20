require('@testing-library/jest-dom');

// Mock vscode.Uri.file
// Provide a minimal mock for vscode API
jest.mock('vscode', () => {
    const originalModule = jest.requireActual('vscode');
    return {
        ...originalModule,
        Uri: {
            ...originalModule.Uri,
            file: jest.fn((path) => ({ fsPath: path })),
        },
        ExtensionMode: {
            Production: 0,
            Development: 1,
            Test: 2,
        },
        ViewColumn: {
            One: 1,
        },
        commands: {
            executeCommand: jest.fn(),
        },
        window: {
            ...originalModule.window,
            showInformationMessage: jest.fn(),
            createWebviewPanel: jest.fn(),
        },
    };
});
