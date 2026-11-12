import type { Metadata } from "next";
import SiteShell from "../components/site-shell";
import "./globals.css";

const description = "A weather prediction-market research project. Explore model equations, historical aggregate validation scores, and fully synthetic examples of signals, trades and payouts.";
export const metadata: Metadata = {
  title: { default: "DailyWeather · Research demo", template: "%s · DailyWeather" },
  description,
  authors: [{ name: "Ethan Wei" }],
  icons: { icon: "/favicon.svg" },
  openGraph: { type: "website", title: "DailyWeather · Research demo", description },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><SiteShell>{children}</SiteShell></body></html>;
}
