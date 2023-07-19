import asyncio

async def foo(semaphore):
    async with semaphore:
        print("Running foo()")
        await asyncio.sleep(2)
        print("Finished foo()")

async def main():
    semaphore = asyncio.Semaphore(4)  # Limiting to 4 concurrent instances
    coroutines = [foo(semaphore) for _ in range(10)]
    print('Gathering...')
    await asyncio.gather(*coroutines)
    print('Gathered')

asyncio.run(main())