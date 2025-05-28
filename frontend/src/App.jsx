import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Portfolio from './pages/Portfolio';
import PortfolioManagement from './pages/PortfolioManagement';
import Transactions from './pages/Transactions';
import Prices from './pages/Prices';
import Risk from './pages/Risk';
import AssetRisk from './pages/AssetRisk';
import Ideas from './pages/Ideas';
import HistoricalChart from './pages/Chart';
import { ToastProvider } from './components/ui/ToastProvider';

function App() {
  return (
    <ToastProvider>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/portfolio" replace />} />
          {/*<Route path="portfolio" element={<Portfolio />} />*/}
          <Route path="portfolio" element={<PortfolioManagement />} />
          <Route path="transactions" element={<Transactions />} />
          <Route path="chart" element={<HistoricalChart />} />
          <Route path="prices" element={<Prices />} />
          <Route path="risk" element={<Risk />} />
          <Route path="/risk/asset" element={<AssetRisk />} />
          <Route path="ideas" element={<Ideas />} />
        </Route>
      </Routes>
    </ToastProvider>
  );
}

export default App;
