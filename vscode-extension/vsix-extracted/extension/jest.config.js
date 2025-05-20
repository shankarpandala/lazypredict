module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  moduleFileExtensions: ['ts', 'tsx', 'js'],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
  },
  testMatch: ['**/__tests__/**/*.test.(ts|tsx|js)'],
  collectCoverage: true,
  moduleNameMapper: {
    '^vscode$': '<rootDir>/__mocks__/vscode.js',
    '^vscode/(.*)$': '<rootDir>/__mocks__/vscode.js',
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  transformIgnorePatterns: [
    'node_modules/(?!(sinon)/)',
  ],
};
