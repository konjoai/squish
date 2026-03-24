"""squish/api — Public REST API layer for Squish.

This package provides stable, versioned REST endpoints following the OpenAI
API schema.  The primary entry point is ``v1_router.register_v1_routes(app)``
which mounts all ``/v1/*`` endpoints onto an ASGI/WSGI application.
"""
