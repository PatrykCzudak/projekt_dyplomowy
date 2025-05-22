import { useState, useEffect } from 'react';
import { api } from '../services/api';

export default function Transactions() {
  const [transactions, setTransactions] = useState([]);
  const [asset, setAsset] = useState('');
  const [type, setType] = useState('BUY');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [total, setTotal] = useState('');

  const fetchTransactions = () => {
    api.get('/transactions/')
      .then(res => setTransactions(res.data))
      .catch(err => console.error('Failed to fetch transactions:', err));
  };

  useEffect(() => {
    fetchTransactions();
  }, []);

  useEffect(() => {
    if (asset.trim()) {
      api.get(`/assets/${asset}/history`)
        .then(res => {
          const history = res.data.historical_prices;
          const latestPrice = Object.values(history).pop();
          setPrice(latestPrice.toFixed(2));
        })
        .catch(() => setPrice(''));
    }
  }, [asset]);

  useEffect(() => {
    const q = parseFloat(quantity);
    const p = parseFloat(price);
    if (!isNaN(q) && !isNaN(p)) {
      setTotal((q * p).toFixed(2));
    } else {
      setTotal('');
    }
  }, [quantity, price]);

  const handleAddTransaction = (e) => {
    e.preventDefault();
    const newTx = {
      symbol: asset,
      type: type.toUpperCase(),
      quantity: parseFloat(quantity),
      price: parseFloat(price)
    };
    api.post('/transactions/', newTx)
      .then(() => {
        setAsset('');
        setType('BUY');
        setQuantity('');
        setPrice('');
        setTotal('');
        fetchTransactions();
      })
      .catch(err => console.error('Failed to add transaction:', err));
  };

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Transactions</h1>
      <h2 className="text-xl font-semibold mb-2">Transaction History</h2>
      <div className="bg-gray-800 rounded-lg shadow p-4 mb-6">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="py-2 px-4 text-left text-xs font-semibold text-gray-400 uppercase">Date</th>
                <th className="py-2 px-4 text-left text-xs font-semibold text-gray-400 uppercase">Asset</th>
                <th className="py-2 px-4 text-left text-xs font-semibold text-gray-400 uppercase">Type</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase">Quantity</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase">Price</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase">Total</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map(tx => {
                const total = tx.quantity * tx.price;
                return (
                  <tr key={tx.id} className="border-b border-gray-800 last:border-0">
                    <td className="py-2 px-4 text-sm">{tx.timestamp ? (new Date(tx.timestamp)).toLocaleDateString() : '-'}</td>
                    <td className="py-2 px-4 text-sm">{tx.symbol} <span className="text-gray-400 text-xs hidden sm:inline">{tx.name}</span></td>
                    <td className="py-2 px-4 text-sm">
                      {tx.type === 'BUY' ? (
                        <span className="text-green-500 font-medium">Buy</span>
                      ) : (
                        <span className="text-red-500 font-medium">Sell</span>
                      )}
                    </td>
                    <td className="py-2 px-4 text-sm text-right">{tx.quantity}</td>
                    <td className="py-2 px-4 text-sm text-right">${tx.price.toFixed(2)}</td>
                    <td className="py-2 px-4 text-sm text-right">${total.toFixed(2)}</td>
                  </tr>
                );
              })}
              {transactions.length === 0 && (
                <tr>
                  <td colSpan="6" className="py-4 px-4 text-center text-gray-500">
                    No transactions found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <h2 className="text-xl font-semibold mb-3">Add New Transaction</h2>
      <div className="bg-gray-800 rounded-lg shadow p-4">
        <form onSubmit={handleAddTransaction}>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex flex-col">
              <label className="text-sm text-gray-300 mb-1">Asset</label>
              <input 
                type="text" 
                value={asset} 
                onChange={(e) => setAsset(e.target.value.toUpperCase())} 
                required 
                placeholder="e.g. AAPL" 
                className="bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-gray-100 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex flex-col">
              <label className="text-sm text-gray-300 mb-1">Type</label>
              <select 
                value={type} 
                onChange={(e) => setType(e.target.value.toUpperCase())} 
                required 
                className="bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="BUY">Buy</option>
                <option value="SELL">Sell</option>
              </select>
            </div>
            <div className="flex flex-col">
              <label className="text-sm text-gray-300 mb-1">Quantity</label>
              <input 
                type="number" 
                value={quantity} 
                onChange={(e) => setQuantity(e.target.value)} 
                required 
                min="0" step="any" 
                placeholder="e.g. 10" 
                className="bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-gray-100 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex flex-col">
              <label className="text-sm text-gray-300 mb-1">Price (auto)</label>
              <input 
                type="text" 
                value={price} 
                disabled
                readOnly
                className="bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-gray-300 text-sm cursor-not-allowed"
              />
            </div>
          </div>
          {total && (
            <div className="mt-2 text-right text-sm text-gray-300">
              Estimated total value: <span className="text-white font-semibold">${total}</span>
            </div>
          )}
          <div className="mt-4 text-right">
            <button 
              type="submit" 
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2"
            >
              Add Transaction
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
