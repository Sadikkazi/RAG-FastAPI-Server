# RAG-FastAPI-Server 🚀

![RAG-FastAPI-Server](https://img.shields.io/badge/RAG-FastAPI-Server-blue?style=for-the-badge&logo=fastapi)

Welcome to the **RAG-FastAPI-Server** repository! This project provides an API server for managing a lightweight Retrieval-Augmented Generation (RAG) database. It utilizes FastAPI and PostgreSQL, enhanced with pgvector for efficient vector similarity search. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Releases](#releases)

## Features

- **FastAPI**: Enjoy the benefits of FastAPI, including automatic generation of OpenAPI documentation and high performance.
- **PostgreSQL**: Utilize PostgreSQL for reliable data storage and management.
- **pgvector**: Implement vector similarity search with pgvector, enabling efficient retrieval of embeddings.
- **Lightweight**: Designed to be simple and easy to set up, making it suitable for various applications.
- **RESTful API**: Follow REST principles for clean and understandable API design.

## Installation

To set up the RAG-FastAPI-Server on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sadikkazi/RAG-FastAPI-Server.git
   cd RAG-FastAPI-Server
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL**:
   - Ensure you have PostgreSQL installed on your machine.
   - Create a database for the application:
     ```sql
     CREATE DATABASE rag_db;
     ```

5. **Configure the database connection**:
   - Update the `DATABASE_URL` in the `.env` file with your PostgreSQL connection string.

6. **Run the server**:
   ```bash
   uvicorn main:app --reload
   ```

Your server should now be running at `http://127.0.0.1:8000`.

## Usage

After installation, you can start interacting with the API. Here’s a quick overview of how to use the RAG-FastAPI-Server:

- **Access the API documentation**: Visit `http://127.0.0.1:8000/docs` to view the automatically generated API documentation.
- **Send requests**: Use tools like Postman or curl to send requests to the server.

### Example Request

Here’s an example of how to make a request to add an embedding:

```bash
curl -X POST "http://127.0.0.1:8000/embeddings" -H "Content-Type: application/json" -d '{"data": [0.1, 0.2, 0.3]}'
```

## API Endpoints

### Embeddings

- **POST /embeddings**: Add a new embedding.
- **GET /embeddings/{id}**: Retrieve an embedding by ID.
- **GET /embeddings**: List all embeddings.

### Search

- **POST /search**: Perform a similarity search on embeddings.

### Health Check

- **GET /health**: Check the health status of the server.

## Contributing

We welcome contributions! If you want to help improve the RAG-FastAPI-Server, please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out:

- **Author**: Sadik Kazi
- **Email**: sadik.kazi@example.com
- **GitHub**: [Sadikkazi](https://github.com/Sadikkazi)

## Releases

To download the latest version of the RAG-FastAPI-Server, visit the [Releases](https://github.com/Sadikkazi/RAG-FastAPI-Server/releases) section. 

You can find detailed release notes and instructions for each version. Make sure to check it out to stay updated with the latest features and improvements.

## Additional Resources

- **FastAPI Documentation**: [FastAPI](https://fastapi.tiangolo.com/)
- **PostgreSQL Documentation**: [PostgreSQL](https://www.postgresql.org/docs/)
- **pgvector Documentation**: [pgvector](https://github.com/pgvector/pgvector)

## Acknowledgments

We would like to thank the FastAPI and PostgreSQL communities for their contributions and support. Your hard work makes projects like this possible.

---

Feel free to explore the code, and let us know how we can improve this project!