---
**Copilot User Preferences:**
- GitHub username: shankarpandala
- Always use this username for all GitHub-related actions and issue creation.

## Agile & Test-Driven Plan for Building a VS Code Extension UI for LazyPredict

This plan describes an Agile, test-driven approach to build, test, and publish a VS Code extension (in TypeScript) that provides a UI for the LazyPredict Python package. The UI will be based on a Figma design (to be provided). All features will be implemented using TDD: write tests first, then implement features. Work will be tracked as GitHub issues, with status maintained in a markdown table below.


### 1. Requirements & Preparation
- Review the Figma design for the UI and clarify all user flows and features.
- Ensure you have a working Python environment with LazyPredict installed for backend integration.
- Install VS Code, Node.js, and `yo code` (VS Code Extension Generator).
- Familiarize yourself with VS Code extension development and the VS Code API.
- **Create GitHub issues for each major requirement and user story.**


### 2. Project Setup (TDD & Agile)
- **Write initial test cases for project scaffolding and setup.**
- Scaffold a new VS Code extension using the TypeScript template (`yo code`).
- Set up the project structure for webview UI (React or plain TypeScript/HTML/CSS as per design).
- Add dependencies for UI framework (if using React, install `react`, `react-dom`, `@types/react`, etc.).
- Set up a build system (Webpack or Vite) for bundling the webview UI.
- **Create/close GitHub issues for each setup step.**


### 3. UI Implementation (TDD & Agile)
- **For each UI feature:**
  - Write test cases for UI components and flows (using Jest, React Testing Library, or similar).
  - Translate the Figma design into UI components (HTML/React/TypeScript/CSS).
  - Implement the main webview panel that will serve as the LazyPredict UI.
  - Add navigation, forms, and controls as per the design (e.g., dataset upload, model selection, results display).
- **Create/close GitHub issues for each UI component/feature.**


### 4. Backend Integration (TDD & Agile)
- **For each backend feature:**
  - Write test cases for extension commands, backend logic, and webview communication.
  - Implement VS Code extension commands to launch the UI panel.
  - Set up communication between the webview and the extension backend using the VS Code Webview API (`postMessage`/`onDidReceiveMessage`).
  - Implement backend logic to:
    - Run LazyPredict commands (e.g., via Python scripts or Jupyter notebooks).
    - Handle dataset upload, model training, and results retrieval.
    - Return results and logs to the UI.
- **Create/close GitHub issues for each backend feature.**


### 5. Python Environment Handling (TDD & Agile)
- **Write test cases for environment detection and error handling.**
- Detect and use the user's Python environment (use VS Code Python extension APIs if needed).
- Ensure the extension can find and run LazyPredict, and provide helpful error messages if not.
- **Create/close GitHub issues for environment handling.**


### 6. Testing & Debugging (TDD & Agile)
- **Write and maintain unit/integration tests for all features.**
- Test the extension locally in VS Code (`F5` to launch Extension Development Host).
- Validate all UI flows, backend integration, and error handling.
- Add unit and integration tests for extension commands and UI logic.
- **Create/close GitHub issues for testing and bug fixes.**


### 7. Packaging & Publishing (TDD & Agile)
- **Write test cases for packaging and publishing steps.**
- Update `package.json` with extension metadata, commands, activation events, and icon.
- Add README, changelog, and usage instructions.
- Package the extension (`vsce package`).
- Publish to the VS Code Marketplace (`vsce publish`).
- **Create/close GitHub issues for packaging and publishing.**


### 8. Maintenance & Updates (TDD & Agile)
- Monitor user feedback and issues.
- Plan for updates and new features as needed.
- **Create/close GitHub issues for maintenance and enhancements.**


---
## Issue Tracking Table

| Issue # | Title                                               | Status | Description                                                                                                                        |
|---------|-----------------------------------------------------|--------|------------------------------------------------------------------------------------------------------------------------------------|
| 515     | Review Figma design and clarify UI requirements     | Closed | Review the Figma design and break down the UI into components and user flows. Document all requirements and clarify ambiguities.    |
| 516     | Scaffold VS Code extension in vscode-extension dir  | Closed | Use the TypeScript template (`yo code`) to scaffold a new VS Code extension in the vscode-extension directory.                      |
| 517     | Write initial test cases for extension setup (TDD)  | Closed | Write test cases for project scaffolding and setup in the vscode-extension directory. Use Jest or a similar framework for TDD.      |
| 518     | Set up build system and dependencies for webview UI | Closed | Set up a build system (Webpack or Vite) and add dependencies for the chosen UI framework in the vscode-extension directory.         |
| 519     | Write test cases for main webview panel and navigation (TDD) | Closed | Write test cases for the main webview panel and navigation components as per the Figma design. Use Jest and React Testing Library (or similar) in the vscode-extension directory. Follow TDD: implement tests before code. |
| 520     | Implement main webview panel and navigation UI | Closed | Implement the main webview panel and navigation UI components in the vscode-extension directory as per the Figma design. Ensure all tests for these components pass (TDD). |
| 521     | Write test cases for dataset upload form (TDD) | Closed | Write test cases for the dataset upload form component as per the Figma design. Use Jest and React Testing Library (or similar) in the vscode-extension directory. Follow TDD: implement tests before code. |
| 522     | Write test cases for model selection UI (TDD) | Closed | Write test cases for the model selection UI component as per the Figma design. Use Jest and React Testing Library (or similar) in the vscode-extension directory. Follow TDD: implement tests before code. |
| 523     | Write test cases for results display UI (TDD) | Closed   | Write test cases for the results display UI component as per the Figma design. Use Jest and React Testing Library (or similar) in the vscode-extension directory. Follow TDD: implement tests before code. |
| 524     | Implement results display UI (TDD)                | Closed   | Implement the results display UI component in the vscode-extension directory as per the Figma design. Ensure all tests pass (TDD). |
| 525     | Write test cases for backend integration (commands, logic, webview communication) (TDD) | Closed   | Write test cases for VS Code extension commands, backend logic for LazyPredict, and webview-extension communication in the vscode-extension directory. Follow TDD. |
| 526     | Implement backend integration (commands, logic, webview communication) | Closed   | Implement VS Code extension commands, backend logic for LazyPredict, and webview-extension communication in the vscode-extension directory. Ensure all tests pass. |
| 527     | Write test cases for packaging and publishing steps (TDD) | Closed   | Write test cases for the packaging and publishing of the VS Code extension, including validation of required package.json fields and asset availability. |
| 528     | Implement packaging and publishing steps | Closed   | Implement the packaging process for the extension, including README.md, CHANGELOG.md, icon, and proper package.json configuration. Fix issues with vsce and generate .vsix file. |
