import React, { useState, useRef } from 'react';
import PortfolioRisk from './PortfolioRisk';
import AssetRisk from './AssetRisk';
import Button from '../components/ui/Button';

export default function Risk() {
  const [tab, setTab] = useState('portfolio');
  const portfolioRef = useRef();
  const assetRef     = useRef();

  const runAnalysis = (mode) => {
    if (tab === 'portfolio') {
      portfolioRef.current?.addAnalysis(mode);
    } else {
      assetRef.current?.addAnalysis(mode);
    }
  };

  return (
    <div className="p-6 bg-surface rounded-lg shadow">
      <h1 className="text-2xl font-bold mb-4">Risk Analysis</h1>

      {/* Tabs */}
      <div className="border-b border-gray-700 mb-6">
        <nav className="-mb-px flex space-x-4">
          <button
            onClick={() => setTab('portfolio')}
            className={`py-2 px-4 font-medium ${
              tab === 'portfolio'
                ? 'border-b-2 border-primary text-primary'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Portfolio Risk
          </button>
          <button
            onClick={() => setTab('asset')}
            className={`py-2 px-4 font-medium ${
              tab === 'asset'
                ? 'border-b-2 border-primary text-primary'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Asset Risk
          </button>
        </nav>
      </div>

      {/* Action buttons */}
      <div className="flex space-x-4 mb-8">
        <Button onClick={() => runAnalysis('Classical')}>
          Calculate Classic Risk
        </Button>
        <Button
          className="bg-success hover:bg-success/90"
          onClick={() => runAnalysis('AI')}
        >
          Calculate AI Risk
        </Button>
      </div>

      {/* Content */}
      <div>
        {tab === 'portfolio'
          ? <PortfolioRisk ref={portfolioRef} />
          : <AssetRisk ref={assetRef} />
        }
      </div>
    </div>
  );
}
