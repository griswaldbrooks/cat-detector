import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Guides',
      items: ['deployment', 'deployment-options'],
    },
    {
      type: 'category',
      label: 'Technical',
      items: ['clip-detection', 'model-evaluation'],
    },
    'claude-code-guide',
  ],
};

export default sidebars;
