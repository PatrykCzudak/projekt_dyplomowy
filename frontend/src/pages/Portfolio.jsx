import { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { api } from '../services/api';

export default function Portfolio() {
  const [portfolio, setPortfolio] = useState([]);

  useEffect(() => {
    api.get('/portfolio/')
      .then(res => {
        setPortfolio(res.data);
      })
      .catch(err => {
        console.error('Failed to fetch portfolio data:', err);
      });
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-4">Portfolio</h1>

      <div className="bg-gray-800 rounded-lg shadow p-4">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="py-2 px-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Asset</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Quantity</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Price</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Value</th>
                <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase tracking-wider">Change</th>
              </tr>
            </thead>
            <tbody>
              {portfolio.map((item, idx) => {
                const value = item.total_quantity * item.current_price;

                return (
                  <tr key={item.symbol || idx} className={idx % 2 !== 0 ? "bg-gray-900" : ""}>
                    <td className="py-2 px-4 text-sm">
                      <span className="font-medium text-gray-100">{item.symbol}</span>
                      {item.name && (
                        <span className="ml-2 text-gray-400 text-xs hidden sm:inline">{item.name}</span>
                      )}
                    </td>
                    <td className="py-2 px-4 text-sm text-right">
                      {item.total_quantity}
                    </td>
                    <td className="py-2 px-4 text-sm text-right">
                      ${item.average_price.toFixed(2)} → <b>${item.current_price.toFixed(2)}</b>
                    </td>
                    <td className="py-2 px-4 text-sm text-right">
                      ${value.toFixed(2)}
                    </td>
                    <td className="py-2 px-4 text-sm text-right">
                      {item.change >= 0 ? (
                        <span className="text-green-500 flex items-center justify-end">
                          <TrendingUp className="w-4 h-4 mr-1" />
                          +{item.change.toFixed(2)}%
                        </span>
                      ) : (
                        <span className="text-red-500 flex items-center justify-end">
                          <TrendingDown className="w-4 h-4 mr-1" />
                          {item.change.toFixed(2)}%
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {portfolio.length === 0 && (
                <tr>
                  <td colSpan="5" className="py-4 px-4 text-center text-gray-500">
                    No data available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
