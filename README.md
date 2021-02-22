# Daily-Algorithm-Practice
[02-06-2021: 给定 100G 的 URL 磁盘数据，使用最多 1G 内存，统计出现频率最高的 Top K 个 URL](#02-06-2021-给定-100g-的-url-磁盘数据使用最多-1g-内存统计出现频率最高的-top-k-个-url)    
[02-07-2021: 10亿个数中如何高效地找到最大的一个数以及最大的第K个数](#02-07-2021-10亿个数中如何高效地找到最大的一个数以及最大的第k个数)  
[02-07-2021: 合并两个有序链表](#02-07-2021合并两个有序链表-leetcode)  
[02-07-2021: 64匹马，8个赛道，找出前4匹马最少需要比几次](#02-07-202164匹马8个赛道找出前4匹马最少需要比几次)  
[02-07-2021: 实现快速排序](#02-07-2021-实现快速排序)  
[02-08-2021: 两数相加 II (Leetcode 445)](#02-08-2021445-两数相加-ii)  
[02-08-2021: 二叉树的锯齿形层次遍历 (Leetcode)](#02-08-2021二叉树的锯齿形层次遍历)  
[02-09-2021: K个一组翻转链表 (Leetcode 25)](#02-09-2021-k-个一组翻转链表-leetcode-25)  
[02-10-2021: 搜索旋转排序数组 (Leetcode 33)](#02-10-2021-搜索旋转排序数组-leetcode-33)  
[02-11-2021: 判断有环链表的环长度](#02-11-2021-判断有环链表的环长度)  
[02-12-2021: 寻找旋转排序数组中的最小值(数组有重复/无重复元素)](#02-12-2021-寻找旋转排序数组中的最小值数组有重复无重复元素-leetcode-153-154)  
[02-13-2021: 最大子序和(Leetcode 53)](#02-13-2021-最大子序和leetcode-53)  
[02-14-2021: 给定一个 foo 函数，60%的概率返回0，40%的概率返回1，如何利用 foo 函数实现一个 50% 返回 0 的函数？(以及利用均匀硬币产生不等概率)](#02-14-2021-给定一个-foo-函数60的概率返回040的概率返回1如何利用-foo-函数实现一个-50-返回-0-的函数)  
[02-15-2021: 搜索旋转排序数组 II (Leetcode 81)](#02-15-2021-搜索旋转排序数组-ii-leetcode-81)  
[02-16-2021: 最长连续子序列(Leetcode 128)](#02-16-2021-最长连续子序列leetcode-128)  
[02-17-2021: 爬楼梯 (Leetcode 70)](#02-17-2021-爬楼梯-leetcode-70)  
[02-18-2021: 用 Rand7() 实现 Rand10() (Leetcode 470)](#02-18-2021-用-rand7-实现-rand10-leetcode-470)  
[02-19-2021: AVL 树和红黑树有什么区别？](#02-19-2021-avl-树和红黑树有什么区别)  
[02-20-2021: 10亿条数据包括 id，上线时间，下线时间，请绘制每一秒在线人数的曲线图](#02-20-2021-10亿条数据包括-id上线时间下线时间请绘制每一秒在线人数的曲线图)  
[02-21-2021: 路径总和 (Leetcode 112)](#02-21-2021-路径总和-leetcode-112)  
[02-22-2021: 数组中的第 K 个最大元素 (Leetcode 215)](#02-22-2021-数组中的第-k-个最大元素-leetcode-215)  

## 02-06-2021: 给定 100G 的 URL 磁盘数据，使用最多 1G 内存，统计出现频率最高的 Top K 个 URL
1. 新建约100个文件，利用hash(URL) % 100的值，将每条URL映射到对应文件下，保证同一URL必然全部映射到同一文件下。
2. 假定分布比较均匀，则每个文件大小约为1G，可在内存中操作，在内存中简单地统计词频即可。
3. 对每个文件都取出相应topK的K个<URL，count>根据count降序排列，形成类似一个数组，将100个文件的数组写入同一个文件。
4. 对100个K对进行[K路归并](https://leetcode-cn.com/problems/merge-k-sorted-lists/). 得到K个即可  
*即使100个K对不能都放进内存：取每个数组的一段放入内存，在K路归并步骤中，在这一段已完全被Heap吐出时，再向堆中插入这一段的下一个<URL, count>即可。
## 02-07-2021: 10亿个数中如何高效地找到最大的一个数以及最大的第K个数  
基本同上。假设一个数为几十个byte大小，10亿个数为几十GB级别，不能全部放入内存。  
**1）最基础解法：K路归并**
1. 一次处理内存放得下的数据量，得到最大的K个数（PriorityQueue或quick selection法）
2. 对所有最大的K个数的数组进行K路归并

**2）询问数据是否有取值或分布范围，则可以使用桶排序/计数排序**  
*3）bash自带的sort可排序超过内存大小的文件 
## 02-07-2021：合并两个有序链表 (Leetcode)
[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
简单
## 02-07-2021：64匹马，8个赛道，找出前4匹马最少需要比几次
智力题，11次   
https://blog.csdn.net/star_fighting/article/details/104706155/
## 02-07-2021: 实现快速排序
Partition -> 利用递归，Sort左，Sort右
```Java
public class Solution {
  public int[] quickSort(int[] array) {
    // Write your solution here
    if (array == null || array.length == 0) {
      return array;
    }
    quickSort(array, 0, array.length - 1);
    return array;
  }

  private void quickSort(int[] array, int lo, int hi) {
    if (lo >= hi) {
      return;
    }
    int pivot = partition(array, lo, hi);
    quickSort(array, lo, pivot - 1);
    quickSort(array, pivot + 1, hi);
  }

  private int partition(int[] array, int lo, int hi) {
    int i = lo + 1, j = hi;
    int val = array[lo];
    while(i <= j) {
      if(array[i] <= val) {
        i++;
      } else if(array[j] >= val) {
        j--;
      } else {
        swap(array, i, j);
        i++;
        j--;
      }
    }
    swap(array, lo, j);
    return j;
  }

  private void swap(int[] array, int i, int j) {
    int temp = array[j];
    array[j] = array[i];
    array[i] = temp;
  }
}
```

## 02-08-2021：445. 两数相加 II
[leetcode 445](https://leetcode-cn.com/problems/add-two-numbers-ii/)  
直接将两链表转化为数字再相加会导致数字溢出❌  
还是采用两数相加I的末位开始处理，最后反转链表。
```Java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Deque<Integer> stack1 = new LinkedList<>();
        Deque<Integer> stack2 = new LinkedList<>();
        while(l1 != null) {
            stack1.push(l1.val);
            l1 = l1.next;
        }
        while(l2 != null) {
            stack2.push(l2.val);
            l2 = l2.next;
        }
        int add1 = 0;
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while(!stack1.isEmpty() || !stack2.isEmpty()) {
            int num1 = stack1.isEmpty() ? 0 : stack1.pop();
            int num2 = stack2.isEmpty() ? 0 : stack2.pop();
            int res = num1 + num2 + add1;
            ListNode node = new ListNode(res % 10);
            cur.next = node;
            cur = node;
            add1 = res / 10;
        }
        if (add1 == 1) {
            cur.next = new ListNode(1);
        }
        ListNode newHead = reverse(dummy.next);
        return newHead;
    }

    private ListNode reverse(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverse(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```

## 02-08-2021：二叉树的锯齿形层次遍历
[leetcode 103](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)  
使用Deque,在BFS过程中两头倒腾即可
```Java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerFirst(root);
        int count = 1;
        while(!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < size; i++) {               
                if (count % 2 != 0) {
                    TreeNode cur = deque.pollFirst();
                    list.add(cur.val);
                    if (cur.left != null) {
                        deque.offerLast(cur.left);
                    }
                    if (cur.right != null) {
                        deque.offerLast(cur.right);
                    }
                } else {
                    TreeNode cur = deque.pollLast();
                    list.add(cur.val);
                    if (cur.right != null) {
                        deque.offerFirst(cur.right);
                    }
                    if (cur.left != null) {
                        deque.offerFirst(cur.left);
                    }
                }
            }
            count++;
            res.add(list);
        }
        return res;
    }
}
```
## 02-09-2021: K 个一组翻转链表 (Leetcode 25)
由于只能使用常数的额外空间，不能使用recursion. 所以只能用类似iterative反转链表的方法。  
prev为已处理好的前半部分，当前k个Node需要反转，next为后半段。注意while循环中，需要最后将cur设置为已被反转k个Node这段的Tail，从而进入下一循环
```Java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode cur = dummy;
        while(true) {
            ListNode prev = cur;
            for(int i = 0; i < k; i++) {
                cur = cur.next;
                if (cur == null) {
                    return dummy.next;
                }
            }
            ListNode next = cur.next;
            ListNode oldHead = prev.next;
            cur.next = null;
            ListNode[] res = reverse(oldHead);
            prev.next = res[0];
            res[1].next = next;
            cur = res[1];
        }
    }

    private ListNode[] reverse(ListNode head) {
        ListNode[] res = new ListNode[2];
        res[1] = head;
        ListNode prev = null;
        while(head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        res[0] = prev;
        return res;
    }
}
```
## 02-10-2021: 搜索旋转排序数组 (Leetcode 33)
观察：经过旋转后的数组，任意取一个区间，左半段右半段中至少有一个是递增的。再根据target是否在这半段取值内，即可实现搜索区间减半
```Java
class Solution {
    public int search(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while(lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] >= nums[lo]) {
                if (target >= nums[lo] && target <= nums[mid]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else {
                if (target >= nums[mid] && target <= nums[hi]) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
        return -1;
    }
}
```
## 02-11-2021: 判断有环链表的环长度
判断有环：快慢指针，假设slow指针走了n步的话。可推导出当 n = k * (环长度) 时，两指针都会相遇一次。  
第一次相遇后，第二个指针再走一圈即可得环长度
```Java
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                int cycleLength = getLength(fast, slow);
                System.out.println(cycleLength);
                return true;
            }
        }
        return false;
    }

    private int getLength(ListNode fast, ListNode slow) {
        int count = 0;
        do {
            fast = fast.next;
            count++;
        } while(fast != slow);
        return count;
    }
}
```
## 02-12-2021: 寻找旋转排序数组中的最小值(数组有重复/无重复元素) Leetcode 153, 154
数组无重复元素：153   
递归法，分别求左右两半的最小值。值得注意的是**如何证明复杂度为O(logN)：** T(N) = T(N/2) + O(1) -> 是因为每次对半分区间后，至少有一半是排序数组。
```Java
class Solution {
    public int findMin(int[] nums) {
        return findMin(nums, 0, nums.length - 1);
    }

    private int findMin(int[] nums, int lo, int hi) {
        // first base case: only 2 or 1 elements
        if (lo + 1 >= hi) {
            return Math.min(nums[lo], nums[hi]);
        }
        // second base case: one half must be sorted
        if (nums[lo] < nums[hi]) {
            return nums[lo];
        }
        int mid = lo + (hi - lo) / 2;
        return Math.min(findMin(nums, lo, mid), findMin(nums, mid + 1, hi));
    }
}
```
数组有重复元素：154    
与153代码一样，但我们要注意当nums[lo] == nums[hi]时，不能推断出lo ~ hi是排序的。例子：33413  
所以时间复杂度为O(N)
```Java
class Solution {
    public int findMin(int[] nums) {
        return findMin(nums, 0, nums.length - 1);
    }

    private int findMin(int[] nums, int lo, int hi) {
        // first base case: only 2 or 1 elements
        if (lo + 1 >= hi) {
            return Math.min(nums[lo], nums[hi]);
        }
        // second base case: if we can find a half is sorted
        // if (nums[lo] == nums[hi]), we can't conclude that [lo ~ hi] is sorted
        // 33313
        if (nums[lo] < nums[hi]) {
            return nums[lo];
        }
        int mid = lo + (hi - lo) / 2;
        return Math.min(findMin(nums, lo, mid), findMin(nums, mid + 1, hi));
    }
}
```

## 02-13-2021: 最大子序和(Leetcode 53)
设dp[i]为0~i最大连续子数组和，且必须包含nums[i] 
```Java
class Solution {
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int max = nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for(int i = 1; i < nums.length; i++) {
            if (dp[i - 1] > 0) {
                dp[i] = dp[i - 1] + nums[i];
            } else {
                dp[i] = nums[i];
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```
## 02-14-2021: 给定一个 foo 函数，60%的概率返回0，40%的概率返回1，如何利用 foo 函数实现一个 50% 返回 0 的函数？
分析连续抛出两次硬币的情况，正反面的出现有四种情况，概率依次为：  
(1) 两次均为正面：0.6 * 0.6=0.36  
(2)第一次正面，第二次反面：0.6 * 0.4=0.24  
(3)第一次反面，第二次正面：0.4 * 0.6=0.24  
(4)两次均为反面：0.4 * 0.4=0.16  
可以看到中间两种情况的概率是完全一样的，于是问题的解法就是连续抛两次硬币，如果两次得到的相同则重新抛两次；否则根据第一次（或第二次）的正面反面情况，就可以得到两个概率相等的事件。

```Java
public int coin() {
  while(true) {
    int a = foo();
    int b = foo();
    if (a != b) {
      return a;
    }
  }
 }
```
***
利用均匀硬币产生不等概率  
问题描述：有一枚均匀的硬币，抛出此硬币后，可用foo()表示其结果。已知foo()能返回0和1两个值，其概率均为0.5。问怎么利用foo()得到另一个函数，使得返回0和1的概率分别为0.3和0.7。  
问题分析：0和1随机生成，可以理解为二进制。可以令a=foo()*2^3+foo()*2^2+foo()*2^1+foo()等概率生成0-15的所有数，只取1~10之间的数，则产生1到10之间的数仍为等概率。%3为0时返回0，%3不为0时返回1
```Java
private int generateNum() {
  while(true) {
    int a = foo() * 2^3 + foo() * 2^2 + foo() * 2^1 + foo();
    if (a >= 1 && a <= 10) {
      return a;
    }
  }    
}

public int coin() {
  int random = generateNum();
  if (random % 3 == 0) {
    return 0;
  } else {
    return 1;
  }
}
```

## 02-15-2021: 搜索旋转排序数组 II (Leetcode 81)
根据nums[mid]与nums[lo]的大小分三种情况讨论：nums[mid] != nums[lo]时候必然可以确认其中半段为递增序列；  
**nums[mid] == nums[lo]时，我们无法得知应该往左半段还是右半段搜索，从而无法舍弃掉任何一半段，只能lo++**
```Java
class Solution {
    public boolean search(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while(lo + 1 < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return true;
            } else if (nums[mid] > nums[lo]) {
                if (target >= nums[lo] && target < nums[mid]) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            } else if (nums[mid] < nums[lo]) {
                if (target > nums[mid] && target <= nums[hi]) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            } else {
                // 无法确定某半段是递增序列的时候
                lo++;
            }
        }
        return nums[lo] == target || nums[hi] == target;
    }
}
```

## 02-16-2021: 最长连续子序列(Leetcode 128)
若只想对含x的一个连续序列只统计一次长度，可利用HashSet是否包含x-1
```Java
class Solution {
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for(int n : nums) {
            set.add(n);
        }
        int max = 0;
        for(int n : set) {
            if (set.contains(n - 1)) {
                continue;
            } else {
                int start = n;
                int count = 0;
                while(set.contains(start)) {
                    count++;
                    start++;
                }
                max = Math.max(max, count);
            }
        }
        return max;
    }
}
```

## 02-17-2021: 爬楼梯 (Leetcode 70)
DP，递推关系：考虑最后一步是爬了一阶还是两阶
```Java
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```

## 02-18-2021: 用 Rand7() 实现 Rand10() (Leetcode 470)
拒绝采样：做两次Rand7(), 使该有序对一一映射到1-49（等概率），取其中的1-40使之实现Rand10().  
其中调用Rand7()的数学期望为：1 * 2 * 40/49 + 2 * 2 * (9/49 * 40/49) + 3 * 2 * (9/49 * 9/49 * 40/49) + ... 等差乘等比数列求和
```Java
class Solution extends SolBase {
    public int rand10() {
        int num;
        do {
            int a = rand7();
            int b = rand7();
            num = (a - 1) * 7 + b;
        } while (num > 40);
        return num % 10 == 0 ? 10 : num % 10;
    }
}
```

## 02-19-2021: AVL 树和红黑树有什么区别？
AVL树是高度平衡的二叉树，平衡条件必须满足（所有节点的左右子树高度差不超过1）不管我们是执行插入还是删除操作，只要不满足上面的条件，就要通过旋转来保持平衡，而的由于旋转比较耗时，由此我们可以知道AVL树适合用于插入与删除次数比较少，但查找多的情况。  
红黑树通过对任何一条从根到叶子的路径上各个节点着色的方式的限制，确保没有一条路径会比其它路径长出两倍，因此，红黑树是一种弱平衡二叉树（由于是弱平衡，可以看到，在相同的节点情况下，AVL树的高度<=红黑树），相对于要求严格的AVL树来说，它的旋转次数少，所以对于搜索，插入，删除操作较多的情况下，用红黑树。**红黑树查找效率也为logN**

## 02-20-2021: 10亿条数据包括 id，上线时间，下线时间，请绘制每一秒在线人数的曲线图
假设做一天内的曲线图，从第0秒开始上线。 那么用长度为86400的int型数组即可表示。(Integer.MAX_VALUE = 2^31 - 1 > 2 * 10^9 > 10亿, int足够表示每秒在线人数)  
定义一个长度为86400的整数数组delta[86400]，每个整数对应这一秒的人数变化值，可能为正也可能为负。开始时将数组元素都初始化为0。  
然后依次读入每个用户的登录时间和退出时间，将与登录时间对应的整数值加1，将与退出时间对应的整数值减1。这样处理一遍后数组中存储了每秒中的人数变化情况。  
定义另外一个长度为86400的整数数组online_num[86400]，每个整数对应这一秒的在线人数。  
假设一天开始时论坛在线人数为0，则第1秒的人数online_num[0]=delta[0]。第n+1秒的人数online_num[n]=online_num[n-1]+delta[n]。  
这样我们就获得了一天中任意时间的在线人数。

## 02-21-2021: 路径总和 (Leetcode 112)
根节点到叶子路径类型的DFS：往下传递时带上当前的值可以解决所有问题。本题简单的带上一个"根到当前节点路径之和"就可解决  
```Java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        return dfs(root, 0, targetSum);
    }

    private boolean dfs(TreeNode root, int curSum, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return curSum + root.val == targetSum;
        }
        return dfs(root.left, curSum + root.val, targetSum) || dfs(root.right, curSum + root.val, targetSum);
    }
}
```

## 02-22-2021: 数组中的第 K 个最大元素 (Leetcode 215)
方法一：使用最小堆，遍历数组时维持当前最大的K个元素。O(nlogk)  
若考虑Heapify -> O(k) + O((n-k)*logk)
```Java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        for(int i = 0; i < nums.length; i++) {
            heap.offer(nums[i]);
            if (i >= k) {
                heap.poll();
            }
        }
        return heap.peek();
    }
}
```
方法二：quick selection. average case: O(N), worse case: O(N^2)  
据说当partition开始前交换第一个元素与后面随机一个元素，可以证明时间复杂度的期望是O(N)
```Java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        return find(nums, k, 0, nums.length - 1);    
    }

    private int find(int[] nums, int k, int lo, int hi) {
        if (lo == hi) {
            return nums[lo];
        }
        int i = lo + 1, j = hi;
        while(i <= j) {
            if (nums[i] <= nums[lo]) {
                i++;
            } else if (nums[j] >= nums[lo]) {
                j--;
            } else {
                swap(nums, i, j);
                i++;
                j--;
            }
        }
        swap(nums, lo, j);
        if(hi - j + 1 == k) {
            return nums[j];
        } else if (hi - j + 1 > k) {
            return find(nums, k, j + 1, hi);
        } else {
            return find(nums, k - hi + j - 1, lo, j - 1);
        }
    }

    private void swap(int[] nums, int lo, int hi) {
        int temp = nums[lo];
        nums[lo] = nums[hi];
        nums[hi] = temp;
    }
}
```
