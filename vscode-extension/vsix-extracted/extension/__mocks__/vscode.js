module.exports = {
  // Window API
  window: {
    showInformationMessage: jest.fn(() => Promise.resolve()),
    showErrorMessage: jest.fn(() => Promise.resolve()),
    showOpenDialog: jest.fn(() => Promise.resolve([{ fsPath: '/mock/path.csv' }])),
    showQuickPick: jest.fn(() => Promise.resolve('col1')),
    showInputBox: jest.fn(() => Promise.resolve('col1')),
    createWebviewPanel: jest.fn(() => ({
      webview: {
        postMessage: jest.fn(),
        onDidReceiveMessage: jest.fn(),
        asWebviewUri: jest.fn((uri) => uri),
      },
      onDidDispose: jest.fn(),
      webviewPanel: null,
    })),
    withProgress: jest.fn((options, task) => task({ report: () => {} }, { onCancellationRequested: () => {} })),
  },
  // Commands API
  commands: {
    registerCommand: ((cmd, callback) => {
        // Store callback for executeCommand
        module.exports._commandCallbacks = module.exports._commandCallbacks || {};
        module.exports._commandCallbacks[cmd] = callback;
        return { dispose: () => {} };
    }),
    executeCommand: ((cmd, ...args) => {
        const cbs = module.exports._commandCallbacks || {};
        const callback = cbs[cmd];
        if (callback) {
            return Promise.resolve(callback(...args));
        }
        return Promise.resolve();
    }),
  },
  // URI API
  Uri: {
    file: jest.fn((p) => ({ fsPath: p })),
    joinPath: jest.fn((uri, ...paths) => ({ fsPath: [uri.fsPath || uri].concat(paths).join('/') })),
    asWebviewUri: jest.fn((uri) => uri),
  },
  // Extension enums
  ViewColumn: { One: 1 },
  ExtensionMode: { Production: 0, Development: 1, Test: 2 },
  ProgressLocation: { Notification: 1 },
  extensions: { getExtension: jest.fn(() => undefined) },
};
