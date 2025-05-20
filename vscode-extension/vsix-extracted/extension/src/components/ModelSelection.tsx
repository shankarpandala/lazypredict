import React from 'react';

interface Model {
  name: string;
  id: string;
}

interface ModelSelectionProps {
  models: Model[];
  selectedModelId?: string;
  onSelect?: (id: string) => void;
}

const ModelSelection: React.FC<ModelSelectionProps> = ({ models, selectedModelId, onSelect }) => {
  return (
    <div>
      <h3>Select Model</h3>
      <ul>
        {models.map(model => (
          <li key={model.id}>
            <label>
              <input
                type="radio"
                name="model"
                value={model.id}
                checked={selectedModelId === model.id}
                onChange={() => onSelect && onSelect(model.id)}
                aria-label={`Select ${model.name}`}
              />
              {model.name}
            </label>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ModelSelection;
