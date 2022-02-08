import arg_services_helper
import grpc
import typer
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc

from arguequery.server import RetrievalService

app = typer.Typer()


def add_services(server: grpc.Server):
    """Add the services to the grpc server."""

    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
        RetrievalService(), server
    )


@app.command()
def main(host: str, port: int):
    """Main entry point for the server."""

    arg_services_helper.serve(
        host,
        port,
        add_services,
        reflection_services=[
            arg_services_helper.full_service_name(retrieval_pb2, "RetrievalService"),
        ],
    )


if __name__ == "__main__":
    app()
