import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ModelSelection from '../src/components/ModelSelection';

describe('ModelSelection', () => {
  const models = [
    { name: 'Random Forest', id: 'rf' },
    { name: 'XGBoost', id: 'xgb' },
    { name: 'Linear Regression', id: 'lr' },
  ];

  it('renders a list of available models', () => {
    render(<ModelSelection models={models} />);
    expect(screen.getByText('Random Forest')).toBeInTheDocument();
    expect(screen.getByText('XGBoost')).toBeInTheDocument();
    expect(screen.getByText('Linear Regression')).toBeInTheDocument();
  });

  it('allows selecting a model', () => {
    const handleSelect = jest.fn();
    render(<ModelSelection models={models} onSelect={handleSelect} />);
    fireEvent.click(screen.getByLabelText('Select Random Forest'));
    expect(handleSelect).toHaveBeenCalledWith('rf');
  });

  it('shows selected model as checked', () => {
    render(<ModelSelection models={models} selectedModelId="xgb" />);
    const radio = screen.getByLabelText('Select XGBoost');
    expect(radio).toBeChecked();
  });
});
