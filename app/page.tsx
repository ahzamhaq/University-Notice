import SearchNotices from "@/components/SearchNotices";

export default function Home() {
  return (
    <main className="container">
      <header className="hero">
        <h1>IPU Notice Search</h1>
        <p>Ask a question and get an answer drawn from GGSIPU notices, with links to the original PDFs.</p>
      </header>
      <SearchNotices />
    </main>
  );
}
