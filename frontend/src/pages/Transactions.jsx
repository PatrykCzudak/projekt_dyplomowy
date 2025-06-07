import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import Spinner from '../components/ui/Spinner';
import { useToast } from '../components/ui/ToastProvider';
import TransactionForm from './TransactionForm';

export default function Transactions() {
  const [transactions, setTransactions] = useState([]);
  const [loadingTxs, setLoadingTxs] = useState(true);
  const showToast = useToast();

  const fetchTransactions = () => {
    setLoadingTxs(true);
    api.get('/transactions/')
      .then(res => setTransactions(res.data))
      .catch(() => showToast('Failed to load transaction history.', 'error'))
      .finally(() => setLoadingTxs(false));
  };

  useEffect(() => {
    fetchTransactions();
  }, []);

  const handleTransactionAdded = newTx => {
    setTransactions(prev => [newTx, ...prev]);
  };

  const SKELETON_ROWS = 5;

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Transaction History</h1>

      <div className="bg-surface rounded-lg shadow p-4">
        <table className="min-w-full border-collapse">
          <thead>
            <tr className="bg-background border-b">
              {['Date','Asset','Type','Quantity','Price','Total'].map(h => (
                <th key={h} className="py-2 px-4 text-left text-xs font-semibold text-gray-600 uppercase">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loadingTxs
              ? Array.from({ length: SKELETON_ROWS }).map((_, i) => (
                  <tr key={i} className="border-b">
                    {Array.from({ length: 6 }).map((__, j) => (
                      <td key={j} className="py-3 px-4">
                        <div className="h-4 bg-gray-700 rounded animate-pulse" />
                      </td>
                    ))}
                  </tr>
                ))
              : transactions.length > 0
                ? transactions.map(tx => {
                    const total = tx.quantity * tx.price;
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
                        <td className="py-2 px-4 text-sm text-right">${total.toFixed(2)}</td>
                      </tr>
                    );
                  })
                : (
                  <tr>
                    <td colSpan="6" className="py-4 px-4 text-center text-gray-500">
                      No Transactions.
                    </td>
                  </tr>
                )
            }
          </tbody>
        </table>
      </div>
    </div>
  );
}
