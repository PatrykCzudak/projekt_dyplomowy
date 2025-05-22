import React, { useState } from "react";
import PortfolioRisk from "./PortfolioRisk";
import AssetRisk from "./AssetRisk";

export default function Risk() {
  const [tab, setTab] = useState("portfolio");

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Risk Analysis</h1>
      <div className="space-x-4 mb-4">
        <button onClick={() => setTab("portfolio")}>Portfolio Risk</button>
        <button onClick={() => setTab("asset")}>Asset Risk</button>
      </div>
      {tab === "portfolio" ? <PortfolioRisk /> : <AssetRisk />}
    </div>
  );
}
