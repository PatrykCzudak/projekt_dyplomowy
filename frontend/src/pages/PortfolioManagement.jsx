import { useState } from 'react';
import HistoricalChart from './Chart';
import TransactionForm from './TransactionForm';
import PortfolioTable from './Portfolio';

export default function PortfolioManagement() {
  const [selectedAsset, setSelectedAsset] = useState('');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleTransactionAdded = newTx => {
    setSelectedAsset(newTx.symbol);
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Portfolio Management</h1>

      <div className="flex flex-col md:flex-row gap-6">
        {/*Wykres*/}
        <div className="md:w-1/2 bg-surface rounded-lg shadow p-4">
          <HistoricalChart
            symbol={selectedAsset}
            onSelectSymbol={setSelectedAsset}
          />
        </div>

        {/*Formularz transakcji*/}
        <div className="md:w-1/2 bg-surface rounded-lg shadow p-4">
          <TransactionForm
            defaultSymbol={selectedAsset}
            onTransactionAdded={handleTransactionAdded}
          />
        </div>
      </div>

      {/*Tabela portfela*/}
      <div className="bg-surface rounded-lg shadow p-4">
        <PortfolioTable
          refreshTrigger={refreshTrigger}
          onRowClick={tx => setSelectedAsset(tx.symbol)}
        />
      </div>
    </div>
  );
}
