import unittest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
import main 

class MockRetriever(BaseRetriever):
    """Mock Retriever that simulates a real retriever's behavior."""
    def _get_relevant_documents(self, query):
        return [Document(page_content="Evaluation frameworks are crucial for AI models.")]

    async def _aget_relevant_documents(self, query):
        return self._get_relevant_documents(query)

class TestRAGPipeline(unittest.TestCase):

    @patch("main.load_pdf")
    @patch("main.create_faiss_vector_store")
    def test_rag_pipeline(self, mock_create_faiss, mock_load_pdf):
        """Test the RAG pipeline end-to-end with mocks."""

        # Mock the document loader
        mock_load_pdf.return_value = [Document(page_content="Evaluation frameworks are crucial for AI models.")]

        # Mock FAISS Vector Store and Retriever
        mock_retriever = MockRetriever()
        mock_create_faiss.return_value = MagicMock(as_retriever=MagicMock(return_value=mock_retriever))

        # Run the pipeline with a mock query
        response = main.rag_pipeline("https://services.google.com/fh/files/misc/evaluation_framework.pdf", "What is the main topic?")

        # Assertions
        self.assertIsNotNone(response)
        self.assertIn("Evaluation frameworks", response)

    def test_split_text(self):
        """Test if text splitter correctly splits text into chunks."""
        documents = [Document(page_content="This is a sample document with multiple lines of text.")]
        chunks = main.split_text(documents, chunk_size=10, chunk_overlap=2)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, Document) for chunk in chunks))

if __name__ == "__main__":
    unittest.main()
