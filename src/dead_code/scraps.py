"""
pickled_path = data_path / 'pickled'
from langchain.document_loaders import PyPDFLoader

pdf_file = dataset_path / '2306.17766.pdf'
pickled_file = (pickled_path / pdf_file.name).with_suffix('.pickle')
if pickled_file.exists():
    with open(pickled_file, 'br') as pickled:
        pages = pickle.load(pickled)
else:
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    with open(pickled_file, 'wb') as pickled:
        pickle.dump(pages, pickled, pickle.HIGHEST_PROTOCOL)
"""

import asyncio


async def foo():
    await asyncio.sleep(1)  # Simulating some asynchronous task


async def main():
    tasks = []
    max_concurrent_tasks = 10

    # Create a list of tasks
    for _ in range(max_concurrent_tasks):
        tasks.append(asyncio.create_task(foo()))

    # Run the tasks concurrently
    await asyncio.gather(*tasks)


# Run the main function
asyncio.run(main())
