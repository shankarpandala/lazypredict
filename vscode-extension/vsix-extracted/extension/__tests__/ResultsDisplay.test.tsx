import React from 'react';
import { render, screen } from '@testing-library/react';
import ResultsDisplay from '../src/components/ResultsDisplay';

describe('ResultsDisplay', () => {
  const mockResults = {
    models: [
      { name: 'RandomForest', accuracy: 0.92, f1: 0.91, time: '1.2s' },
      { name: 'LinearRegression', accuracy: 0.89, f1: 0.88, time: '0.8s' },
    ],
    bestModel: 'RandomForest',
    summary: 'RandomForest performed best on this dataset.'
  };

  it('renders the results table with model names, accuracy, f1, and times', () => {
    render(<ResultsDisplay results={mockResults} />);
    expect(screen.getByText('RandomForest')).toBeInTheDocument();
    expect(screen.getByText('LinearRegression')).toBeInTheDocument();
    expect(screen.getByText('0.92')).toBeInTheDocument();
    expect(screen.getByText('0.89')).toBeInTheDocument();
    expect(screen.getByText('0.91')).toBeInTheDocument();
    expect(screen.getByText('0.88')).toBeInTheDocument();
    expect(screen.getByText('1.2s')).toBeInTheDocument();
    expect(screen.getByText('0.8s')).toBeInTheDocument();
  });

  it('highlights the best model', () => {
    render(<ResultsDisplay results={mockResults} />);
    const bestModelRow = screen.getByTestId('model-row-RandomForest');
    expect(bestModelRow).toHaveClass('best-model');
  });

  it('shows a summary of the results', () => {
    render(<ResultsDisplay results={mockResults} />);
    expect(screen.getByText('RandomForest performed best on this dataset.')).toBeInTheDocument();
  });

  it('renders a message if no results are available', () => {
    render(<ResultsDisplay results={null} />);
    expect(screen.getByText(/No results to display/i)).toBeInTheDocument();
  });
});
