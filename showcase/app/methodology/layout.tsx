import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Methodology",
  description: "Historical aggregate model validation, the S01 equations and downloadable synthetic trade fixtures.",
};

export default function MethodologyLayout({ children }: { children: React.ReactNode }) {
  return children;
}
