import type { Metadata } from "next";
import ModelLab from "../../components/model-lab";

const description = "Follow a synthetic Tokyo trade from forecast errors and quoted prices to entry rules, a fill and a payout.";

export const metadata: Metadata = {
  title: "Model & Trader",
  description,
};

export default function ModelPage() {
  return (
    <>
      <header className="page-heading">
        <div>
          <h1>Model &amp; Trader</h1>
          <p>Synthetic Tokyo trade. The diagram applies the S01 rules to fictional forecast probabilities, quotes and cash flows.</p>
        </div>
      </header>
      <ModelLab />
    </>
  );
}
