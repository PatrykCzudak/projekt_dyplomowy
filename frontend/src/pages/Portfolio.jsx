import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { api } from '../services/api';

export default function PortfolioTable({ refreshTrigger, onRowClick }) {
  const [portfolio, setPortfolio] = useState([]);

  useEffect(() => {
    api.get('/portfolio/')
       .then(res => setPortfolio(res.data))
       .catch(err => console.error('Failed to fetch portfolio data:', err));
  }, [refreshTrigger]);

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Portfolio Overview</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="py-2 px-4 text-left text-xs font-semibold text-gray-400 uppercase">Asset</th>
              <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase">Quantity</th>
              <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase">Price</th>
              <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase">Value</th>
              <th className="py-2 px-4 text-right text-xs font-semibold text-gray-400 uppercase">Change</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.length > 0 ? (
              portfolio.map((item, idx) => {
                const value = item.total_quantity * item.current_price;
                // procentowa zmiana względem średniej ceny zakupu
                const percentChange = ((item.current_price - item.average_price) / item.average_price) * 100;
                const formattedPercent = percentChange.toFixed(2);

                return (
                  <tr
                    key={item.symbol || idx}
                    className={idx % 2 !== 0 ? 'bg-gray-900' : ''}
                    onClick={() => onRowClick(item)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td className="py-2 px-4 text-sm">
                      <span className="font-medium text-gray-100">{item.symbol}</span>
                      {item.name && (
                        <span className="ml-2 text-gray-400 text-xs hidden sm:inline">
                          {item.name}
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-4 text-sm text-right">{item.total_quantity}</td>
                    <td className="py-2 px-4 text-sm text-right">
                      ${item.average_price.toFixed(2)} → <b>${item.current_price.toFixed(2)}</b>
                    </td>
                    <td className="py-2 px-4 text-sm text-right">${value.toFixed(2)}</td>
                    <td className="py-2 px-4 text-sm text-right">
                      {percentChange >= 0 ? (
                        <span className="text-green-500 flex items-center justify-end">
                          <TrendingUp className="w-4 h-4 mr-1" />
                          +{formattedPercent}%
                        </span>
                      ) : (
                        <span className="text-red-500 flex items-center justify-end">
                          <TrendingDown className="w-4 h-4 mr-1" />
                          {formattedPercent}%
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })
            ) : (
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
  );
}
