#pragma once

#define BANKSCOUNT 32
#define THREADSCOUNT 2048

static dim3 threadsPerBlock(16, 16, 2);
static int threadsInOneBlock = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;
static int blockCount = (int)ceil(float(THREADSCOUNT) / float(threadsInOneBlock));
static dim3 numBlocks(blockCount);