// Defines the structure for messages passed between the webview and the extension.

/**
 * Messages sent from the Webview to the Extension.
 */
export type WebviewToExtensionMessage = 
    | { command: 'runLazyPredict'; data: { filePath: string; problemType: 'classification' | 'regression'; customModels?: string[] } }
    | { command: 'getInitialData' }
    | { command: 'showInfo'; text: string };

/**
 * Messages sent from the Extension to the Webview.
 */
export type ExtensionToWebviewMessage = 
    | { command: 'analysisResult'; data: any; error?: undefined }
    | { command: 'analysisError'; error: string; data?: undefined }
    | { command: 'initialData'; data: any } // e.g., previously saved settings or state
    | { command: 'showLoading'; isLoading: boolean }
    | { command: 'showNotification'; messageType: 'info' | 'error' | 'warning'; text: string };
