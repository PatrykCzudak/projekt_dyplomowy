import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import Input from '../components/ui/Input';
import Button from '../components/ui/Button';
import Spinner from '../components/ui/Spinner';
import { useToast } from '../components/ui/ToastProvider';

export default function TransactionForm({ defaultSymbol = '', onTransactionAdded }) {
  const [asset, setAsset] = useState(defaultSymbol);
  const [type, setType] = useState('BUY');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [total, setTotal] = useState('');
  const [loading, setLoading] = useState(false);
  const showToast = useToast();

  // 1. pobieramy cenę, gdy zmieni się ticker
  useEffect(() => {
    if (!asset) return;
    api.get(`/assets/${asset}/history`)
      .then(res => {
        const hist = res.data.historical_prices;
        const dates = Object.keys(hist);
        const latest = hist[dates[dates.length - 1]];
        setPrice(latest.toFixed(2));
      })
      .catch(() => setPrice(''));
  }, [asset]);

  // 2. wyliczamy total
  useEffect(() => {
    const q = parseFloat(quantity);
    const p = parseFloat(price);
    setTotal(!isNaN(q) && !isNaN(p) ? (q * p).toFixed(2) : '');
  }, [quantity, price]);

  // 3. obsługa submit
  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    try {
      const payload = {
        symbol: asset,
        type,
        quantity: +quantity,
        price: +price,
      };
      const { data: newTx } = await api.post('/transactions/', payload);
      showToast('Dodano transakcję', 'success');
      onTransactionAdded(newTx);
      // wyczyść formularz
      setQuantity('');
    } catch {
      showToast('Błąd przy dodawaniu transakcji', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-surface rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Add New Transaction</h2>
      <form onSubmit={handleSubmit} className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Asset */}
        <div className="flex flex-col">
          <label className="text-sm text-gray-700 mb-1">Asset</label>
          <Input
            type="text"
            placeholder="e.g. AAPL"
            value={asset}
            onChange={e => setAsset(e.target.value.toUpperCase())}
            required
          />
        </div>

        {/* Type–select */}
        <div className="flex flex-col">
          <label className="text-sm text-gray-700 mb-1">Type</label>
          <select
            value={type}
            onChange={e => setType(e.target.value)}
            className="block w-full bg-transparent border-0 border-b-2 border-primary focus:outline-none focus:ring-0 py-2"
          >
            <option value="BUY">Buy</option>
            <option value="SELL">Sell</option>
          </select>
        </div>

        {/* Quantity */}
        <div className="flex flex-col">
          <label className="text-sm text-gray-700 mb-1">Quantity</label>
          <Input
            type="number"
            placeholder="e.g. 10"
            value={quantity}
            onChange={e => setQuantity(e.target.value)}
            required
          />
        </div>

        {/* Price (auto) */}
        <div className="flex flex-col">
          <label className="text-sm text-gray-700 mb-1">Price (auto)</label>
          <Input
            type="text"
            value={price}
            disabled
            readOnly
          />
        </div>

        {/* Submit */}
        <div className="col-span-full">
          <Button
            loading={loading}
            type="submit"
            className="
                w-full py-3 text-lg uppercase tracking-wide
                border-2 border-white
                bg-gradient-to-r from-primary to-secondary
                hover:from-secondary hover:to-primary
                rounded-lg shadow-md
                transform hover:scale-105
                transition duration-200
                focus:outline-none focus:ring-4 focus:ring-primary/50
              "
          >
            Add
          </Button>
        </div>
      </form>
      {total && (
        <div className="mt-6 text-right">
          <span className="text-2xl font-bold text-success">
            Total: ${total}
          </span>
        </div>
      )}
    </div>
  );
}
