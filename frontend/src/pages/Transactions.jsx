import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import Spinner from '../components/ui/Spinner';
import { useToast } from '../components/ui/ToastProvider';

export default function Transactions() {
  const [transactions, setTransactions] = useState([]);
  const [loadingTxs, setLoadingTxs] = useState(true);
  const showToast = useToast();

  const fetchTransactions = () => {
    setLoadingTxs(true);
    api
      .get('/transactions/')
      .then(res => setTransactions(res.data))
      .catch(() => showToast('Nie udało się załadować historii transakcji', 'error'))
      .finally(() => setLoadingTxs(false));
  };

  useEffect(() => {
    fetchTransactions();
  }, []);

  const handleTransactionAdded = newTx => {
    setTransactions(prev => [newTx, ...prev]);
  };

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Transaction History</h1>
      {/* history table */}
      {loadingTxs ? (
        <div className="flex justify-center">
          <Spinner className="w-8 h-8" />
        </div>
      ) : (
        <div className="bg-surface rounded-lg shadow p-4">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-background border-b">
                <th className="py-2 px-4 text-left text-xs font-semibold text-gray-600 uppercase">Date</th>
                <th className="py-2 px-4 text-left text-xs font-semibold text-gray-600 uppercase">Asset</th>
                <th className="py-2 px-4 text-left text-xs font-semibold text-gray-600 uppercase">Type</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-600 uppercase">Quantity</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-600 uppercase">Price</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-600 uppercase">Total</th>
              </tr>
            </thead>
            <tbody>
              {transactions.length > 0 ? (
                transactions.map(tx => {
                  const totalVal = tx.quantity * tx.price;
                  const date = tx.timestamp
                    ? new Date(tx.timestamp).toLocaleDateString()
                    : '-';
                  return (
                    <tr key={tx.id} className="border-b last:border-0">
                      <td className="py-2 px-4 text-sm">{date}</td>
                      <td className="py-2 px-4 text-sm">{tx.symbol}</td>
                      <td className="py-2 px-4 text-sm">
                        <span className={tx.type === 'BUY' ? 'text-green-600' : 'text-red-600'}>
                          {tx.type}
                        </span>
                      </td>
                      <td className="py-2 px-4 text-sm text-right">{tx.quantity}</td>
                      <td className="py-2 px-4 text-sm text-right">${tx.price.toFixed(2)}</td>
                      <td className="py-2 px-4 text-sm text-right">${totalVal.toFixed(2)}</td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="6" className="py-4 px-4 text-center text-gray-500">
                    Brak transakcji.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
