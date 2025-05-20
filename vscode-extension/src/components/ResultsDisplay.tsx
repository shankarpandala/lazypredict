import React from 'react';

export interface ModelResult {
  name: string;
  accuracy: number;
  f1: number;
  time: string;
  [key: string]: any; // Allow other properties
}

export interface ResultsData {
  models: ModelResult[];
  bestModel: string;
  logs?: string; // Optional logs
  summary?: string; // Optional summary, if different from logs
}

export interface ResultsDisplayProps {
  results: ResultsData | null;
  isLoading?: boolean;
  error?: string | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, isLoading = false, error = null }) => {
  if (isLoading) {
    return <div data-testid="results-loading">Loading results...</div>;
  }
  if (error) {
    return <div data-testid="results-error" role="alert" className="error-message">{error}</div>;
  }
  if (!results || !results.models || results.models.length === 0) {
    return <div data-testid="results-empty">No results to display. Please run the analysis.</div>;
  }

  const { models, bestModel, logs, summary } = results;

  // Determine columns dynamically but ensure key ones are present
  // For simplicity, let's assume a fixed set of important columns for now,
  // but this could be made more dynamic based on the actual data received.
  const columnOrder: Array<keyof ModelResult> = ['name', 'accuracy', 'f1', 'time'];
  const displayColumns = columnOrder.filter(colKey => models[0] && models[0][colKey] !== undefined);


  return (
    <div data-testid="results-container" className="results-display-container">
      <h2 data-testid="results-header">Results</h2>
      <div data-testid="results-table-container" className="table-responsive">
        <table data-testid="results-table" className="results-table">
          <thead>
            <tr>
              {displayColumns.map((colKey) => (
                <th key={String(colKey)}>{String(colKey).charAt(0).toUpperCase() + String(colKey).slice(1)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr 
                key={model.name} 
                data-testid={`model-row-${model.name}`}
                className={model.name === bestModel ? 'best-model' : ''}
              >
                {displayColumns.map((colKey) => (
                  <td key={`${model.name}-${String(colKey)}`}>{model[colKey]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {summary && (
        <div data-testid="results-summary" className="results-summary">
          <h3>Summary</h3>
          <p>{summary}</p>
        </div>
      )}
      {logs && (
        <div data-testid="results-logs" className="results-logs">
          <h3>Logs</h3>
          <pre>{logs}</pre>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
