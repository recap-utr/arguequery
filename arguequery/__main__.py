import typing as t

import arg_services_helper
import grpc
import typer
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc

from arguequery.config import config
from arguequery.server import RetrievalService
from arguequery.services import nlp

app = typer.Typer()


def add_services(server: grpc.Server):
    """Add the services to the grpc server."""

    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
        RetrievalService(), server
    )


@app.command()
def main(retrieval_addr: t.Optional[str] = None, nlp_address: t.Optional[str] = None):
    """Main entry point for the server."""

    if nlp_address:
        nlp.address = nlp_address
        nlp.client = nlp.init_client()

    arg_services_helper.serve(
        retrieval_addr or config.retrieval_address,
        add_services,
        [arg_services_helper.full_service_name(retrieval_pb2, "RetrievalService")],
    )


if __name__ == "__main__":
    app()
