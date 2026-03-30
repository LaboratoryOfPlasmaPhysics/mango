import uvicorn

from mango.app import create_app

app = create_app()


def main():
    uvicorn.run("mango.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
