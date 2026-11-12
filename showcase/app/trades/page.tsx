import { Suspense } from "react";
import type { Metadata } from "next";
import TradeExplorer from "../../components/trade-explorer";

export const metadata: Metadata = {
  title: "Trade ledger",
  description: "Explore synthetic market positions, from model signals and quotes to example fills and payouts.",
};

export default function TradesPage() {
  return (
    <>
      <header className="page-heading">
        <div>
          <h1>Trade ledger</h1>
          <p>Synthetic trade ledger. Every signal, quote, fill and payout on this page is a fictional example.</p>
        </div>
      </header>
      <Suspense fallback={<p>Loading the synthetic trade ledger…</p>}><TradeExplorer/></Suspense>
    </>
  );
}
