"use client";

import { useEffect, useState } from "react";

import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";

import UploadSection from "@/components/dashboard/UploadSection";
import MetricCards from "@/components/dashboard/MetricCards";
import SummaryCharts from "@/components/dashboard/SummaryCharts";
import NetworkGraph from "@/components/dashboard/NetworkGraph";
import LaunderingGroupsTable from "@/components/dashboard/LaunderingGroupsTable";

import { fetchHiSmallDemo } from "@/lib/api";
import { DashboardResponse } from "@/types/aml";

export default function Home() {
  const [data, setData] = useState<DashboardResponse | null>(null);

  const [loading, setLoading] = useState(true);

  useEffect(() => {
  async function loadDemo() {
    try {
      setLoading(true);

      const result = await fetchHiSmallDemo();

      setData(result);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }

  loadDemo();
}, []);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,#3b2f12_0%,#020617_38%,#000_100%)] text-white flex flex-col">
      <Header />

      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-6 space-y-6">
        <UploadSection />

        {loading ? (
          <div className="text-center py-20">
            Loading AML Dashboard...
          </div>
        ) : (
          <>
            <MetricCards summary={data?.summary} />

            <SummaryCharts
              summary={data?.summary}
              metrics={data?.metrics}
            />

            <NetworkGraph graph={data?.graph} />

            <LaunderingGroupsTable
              groups={data?.graph?.groups ?? []}
            />
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}