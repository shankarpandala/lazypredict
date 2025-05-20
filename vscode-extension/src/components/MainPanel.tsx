import React from 'react';
import Navigation from './Navigation';

const MainPanel: React.FC = () => {
  return (
    <div data-testid="main-panel">
      <Navigation />
      <div>
        <h2>Upload Dataset</h2>
        {/* Placeholder for upload form and other content */}
      </div>
    </div>
  );
};

export default MainPanel;
