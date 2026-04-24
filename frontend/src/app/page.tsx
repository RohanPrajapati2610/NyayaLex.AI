import ChatWindow from "@/components/ChatWindow";
import Header from "@/components/Header";

export default function Home() {
  return (
    <main className="flex flex-col h-screen">
      <Header />
      <ChatWindow />
    </main>
  );
}
