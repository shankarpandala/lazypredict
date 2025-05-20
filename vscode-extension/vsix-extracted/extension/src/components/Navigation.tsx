import React from 'react';

interface NavigationProps {
  selected?: string;
}

const sections = [
  { label: 'Upload', key: 'upload' },
  { label: 'Models', key: 'models' },
  { label: 'Results', key: 'results' },
];

const Navigation: React.FC<NavigationProps> = ({ selected = 'upload' }) => {
  return (
    <nav data-testid="navigation">
      <ul>
        {sections.map(section => (
          <li
            key={section.key}
            className={selected === section.key ? 'selected' : ''}
          >
            {section.label}
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default Navigation;
