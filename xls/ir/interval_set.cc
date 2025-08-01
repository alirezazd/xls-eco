// Copyright 2021 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/ir/interval_set.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/sliding_window.hpp"
#include "xls/common/iterator_range.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"

namespace xls {

IntervalSet IntervalSet::Maximal(int64_t bit_count) {
  IntervalSet result(bit_count);
  result.AddInterval(Interval::Maximal(bit_count));
  result.is_normalized_ = true;
  return result;
}

IntervalSet IntervalSet::NonZero(int64_t bit_count) {
  CHECK_GE(bit_count, 0);
  IntervalSet result(bit_count);
  result.AddInterval(Interval(UBits(1, bit_count), Bits::AllOnes(bit_count)));
  result.is_normalized_ = true;
  return result;
}

IntervalSet IntervalSet::Precise(const Bits& bits) {
  IntervalSet result(bits.bit_count());
  result.AddInterval(Interval::Precise(bits));
  result.is_normalized_ = true;
  return result;
}

IntervalSet IntervalSet::Punctured(const Bits& bits) {
  IntervalSet result(bits.bit_count());
  if (!bits.IsZero()) {
    result.AddInterval(Interval::RightOpen(Bits(bits.bit_count()), bits));
  }
  if (!bits.IsAllOnes()) {
    result.AddInterval(
        Interval::LeftOpen(bits, Bits::AllOnes(bits.bit_count())));
  }
  result.is_normalized_ = true;
  return result;
}

void IntervalSet::SetIntervals(absl::Span<const Interval> intervals) {
  is_normalized_ = false;
  intervals_.clear();
  if (intervals.empty()) {
    bit_count_ = -1;
  } else {
    bit_count_ = intervals[0].BitCount();
  }

  for (const Interval& interval : intervals) {
    AddInterval(interval);
  }
}

void IntervalSet::Normalize() {
  if (is_normalized_) {
    return;
  }

  // Fastpath single proper interval
  if (intervals_.size() == 1 && !intervals_.front().IsImproper()) {
    // A single proper interval is definitionally normalized.
    is_normalized_ = true;
    return;
  }

  Bits zero(BitCount());
  Bits max = Bits::AllOnes(BitCount());
  std::vector<Interval> expand_improper;
  for (const Interval& interval : intervals_) {
    if (interval.IsImproper()) {
      expand_improper.push_back(Interval(zero, interval.UpperBound()));
      expand_improper.push_back(Interval(interval.LowerBound(), max));
    } else {
      expand_improper.push_back(interval);
    }
  }

  std::sort(expand_improper.begin(), expand_improper.end());

  intervals_.clear();
  for (int32_t i = 0; i < expand_improper.size();) {
    Interval interval = expand_improper[i++];
    while ((i < expand_improper.size()) &&
           (Interval::Overlaps(interval, expand_improper[i]) ||
            Interval::Abuts(interval, expand_improper[i]))) {
      interval = Interval::ConvexHull(interval, expand_improper[i]);
      ++i;
    }
    intervals_.push_back(interval);
  }

  is_normalized_ = true;
}

std::optional<Interval> IntervalSet::ConvexHull() const {
  CHECK_GE(bit_count_, 0);
  std::optional<Bits> lower = LowerBound();
  std::optional<Bits> upper = UpperBound();
  if (!lower.has_value() || !upper.has_value()) {
    return std::nullopt;
  }
  return Interval(lower.value(), upper.value());
}

std::optional<Bits> IntervalSet::LowerBound() const {
  CHECK_GE(bit_count_, 0);
  if (is_normalized_) {
    if (intervals_.empty()) {
      return std::nullopt;
    }
    return intervals_.front().LowerBound();
  }

  std::optional<Bits> lower;
  for (const Interval& interval : intervals_) {
    if (lower.has_value()) {
      if (bits_ops::ULessThan(interval.LowerBound(), lower.value())) {
        lower = interval.LowerBound();
      }
    } else {
      lower = interval.LowerBound();
    }
  }
  return lower;
}

std::optional<Bits> IntervalSet::UpperBound() const {
  CHECK_GE(bit_count_, 0);
  if (is_normalized_) {
    if (intervals_.empty()) {
      return std::nullopt;
    }
    return intervals_.back().UpperBound();
  }

  std::optional<Bits> upper;
  for (const Interval& interval : intervals_) {
    if (upper.has_value()) {
      if (bits_ops::UGreaterThan(interval.UpperBound(), upper.value())) {
        upper = interval.UpperBound();
      }
    } else {
      upper = interval.UpperBound();
    }
  }
  return upper;
}

IntervalSet IntervalSet::ZeroExtend(int64_t bit_width) const {
  CHECK_GE(bit_width, BitCount());
  IntervalSet result(bit_width);
  for (const Interval& interval : Intervals()) {
    result.AddInterval(interval.ZeroExtend(bit_width));
  }
  result.Normalize();
  return result;
}

IntervalSet IntervalSet::PositiveIntervals(bool with_zero) const {
  CHECK(IsNormalized());
  if (bit_count_ < 1 || (bit_count_ == 1 && !with_zero)) {
    return IntervalSet(bit_count_);
  }
  return IntervalSet::Intersect(
      *this, IntervalSet::Of({Interval(UBits(with_zero ? 0 : 1, bit_count_),
                                       Bits::MaxSigned(bit_count_))}));
}

IntervalSet IntervalSet::NegativeAbsoluteIntervals() const {
  CHECK(IsNormalized());
  if (bit_count_ < 1) {
    return IntervalSet();
  }
  IntervalSet segment = IntervalSet::Intersect(
      *this, IntervalSet::Of({Interval(Bits::MinSigned(bit_count_),
                                       SBits(-1, bit_count_))}));
  IntervalSet res(bit_count_);
  for (const Interval& interval : segment.Intervals()) {
    res.AddInterval(Interval(bits_ops::Negate(interval.UpperBound()),
                             bits_ops::Negate(interval.LowerBound())));
  }
  res.Normalize();
  return res;
}

/* static */ bool IntervalSet::Disjoint(const IntervalSet& lhs,
                                        const IntervalSet& rhs) {
  CHECK(lhs.IsNormalized() && rhs.IsNormalized());
  // Try the trivial cases first.
  if (lhs.IsEmpty() || rhs.IsEmpty()) {
    return true;
  }
  if (lhs.IsMaximal() || rhs.IsMaximal()) {
    return false;
  }
  Interval lhs_convex = *lhs.ConvexHull();
  Interval rhs_convex = *rhs.ConvexHull();
  if (Interval::Disjoint(lhs_convex, rhs_convex)) {
    return true;
  }
  // Both intervals lists are sorted by lower bound since they are normalized.
  absl::Span<Interval const> left_intervals = lhs.Intervals();
  absl::Span<Interval const> right_intervals = rhs.Intervals();
  auto lhs_it = left_intervals.cbegin();
  auto rhs_it = right_intervals.cbegin();
  while (lhs_it != left_intervals.cend() && rhs_it != right_intervals.cend()) {
    if (Interval::Overlaps(*lhs_it, *rhs_it)) {
      return false;
    }
    if (*lhs_it < *rhs_it) {
      ++lhs_it;
    } else {
      ++rhs_it;
    }
  }
  return true;
}

namespace {
// Precondition: low_it and high_it are both iterators into a sorted interval
// list containing only proper intervals, the list is the minimal
// representation of the interval set, and the lower_bound of *low_it is <= the
// lower_bound of *high_it
//
// This will return the upper bound of the combined interval starting at
// low_it->lower_bound. The low_it will be advanced to the position after the
// last interval subsumed by the interval between the initial
// low_it->lower_bound and the returned value. The high_it will be advanced to
// the position after the last interval subsumed by the interval between the
// initial low_it->lower_bound and the returned value.
//
// This is to optimize a hot-spot in the CondSpec pass.
const Bits& CombineAndAdvance(absl::Span<Interval const>::iterator& low_it,
                              absl::Span<Interval const>::iterator low_end,
                              absl::Span<Interval const>::iterator& high_it,
                              absl::Span<Interval const>::iterator high_end) {
  // Both inputs are normalized so we know that no intervals in low_it or
  // high_it overlap with other intervals in the same list.
  DCHECK_NE(high_it, high_end);
  DCHECK_NE(low_it, low_end);
  // Short circuit contains everything remaining.
  if (low_it->UpperBound().IsAllOnes()) {
    const Bits& res = low_it->UpperBound();
    low_it = low_end;
    high_it = high_end;
    return res;
  }
  if (bits_ops::UGreaterThan(high_it->LowerBound(), low_it->UpperBound())) {
    // high is fully outside of low.
    const Bits& res = low_it->UpperBound();
    ++low_it;
    return res;
  }
  // combines both low and high.
  // Advance past any fully contained intervals.
  while (
      high_it != high_end &&
      bits_ops::ULessThanOrEqual(high_it->UpperBound(), low_it->UpperBound())) {
    ++high_it;
  }
  // 3 possibilities.
  // 1) All the other sets values have been consumed. Means low_it contains
  // all of them. Just return.
  // 2) low and high don't overlap at all, just return.
  if (high_it == high_end ||
      bits_ops::UGreaterThan(high_it->LowerBound(), low_it->UpperBound())) {
    const Bits& res = low_it->UpperBound();
    ++low_it;
    return res;
  }
  // 3) low and high still overlap.
  ++low_it;
  if (low_it != low_end) {
    return CombineAndAdvance(high_it, high_end, low_it, low_end);
  }
  // no more lows.
  const Bits& res = high_it->UpperBound();
  ++high_it;
  return res;
}
}  // namespace

// Combine 2 normalized intervals. This is optimized to improve a hotspot in
// cond-spec.
IntervalSet IntervalSet::Combine(const IntervalSet& lhs,
                                 const IntervalSet& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  if (!lhs.IsNormalized() || !rhs.IsNormalized()) {
    IntervalSet l = lhs;
    l.Normalize();
    IntervalSet r = rhs;
    r.Normalize();
    return Combine(l, r);
  }
  std::vector<Interval> combined_intervals;
  combined_intervals.reserve(lhs.Intervals().size() + rhs.Intervals().size());
  auto it_l = lhs.Intervals().begin();
  auto it_r = rhs.Intervals().begin();
  while (it_l != lhs.Intervals().end() && it_r != rhs.Intervals().end()) {
    bool lt = bits_ops::ULessThan(it_l->LowerBound(), it_r->LowerBound());
    Bits lower;
    Bits upper;
    if (lt) {
      lower = it_l->LowerBound();
      upper = CombineAndAdvance(it_l, lhs.Intervals().end(), it_r,
                                rhs.Intervals().end());
    } else {
      lower = it_r->LowerBound();
      upper = CombineAndAdvance(it_r, rhs.Intervals().end(), it_l,
                                lhs.Intervals().end());
    }
    // This doesn't merge adjacent values so we need to do that ourselves.
    if (!combined_intervals.empty() &&
        bits_ops::Increment(combined_intervals.back().UpperBound()) == lower) {
      combined_intervals.back() =
          Interval(combined_intervals.back().LowerBound(), upper);
    } else {
      combined_intervals.push_back(Interval(lower, upper));
    }
  }
  // Add the rest of the intervals unmodified.
  if (it_l != lhs.Intervals().end() || it_r != rhs.Intervals().end()) {
    bool first = true;
    auto [remaining, end_it] =
        (it_l == lhs.Intervals().end())
            ? std::make_pair(it_r, rhs.Intervals().end())
            : std::make_pair(it_l, lhs.Intervals().end());
    for (; remaining != end_it; ++remaining) {
      if (first && !combined_intervals.empty() &&
          bits_ops::Increment(combined_intervals.back().UpperBound()) ==
              remaining->LowerBound()) {
        combined_intervals.back() = Interval(
            combined_intervals.back().LowerBound(), remaining->UpperBound());
      } else {
        combined_intervals.push_back(*remaining);
      }
      first = false;
    }
  }

  // By construction we know that the result is already normalized.
  return IntervalSet(std::in_place, /*is_normalized=*/true,
                     /*bit_count=*/lhs.BitCount(),
                     /*intervals=*/std::move(combined_intervals));
}

IntervalSet IntervalSet::Intersect(const IntervalSet& lhs,
                                   const IntervalSet& rhs) {
  CHECK_EQ(lhs.BitCount(), rhs.BitCount());
  CHECK(lhs.is_normalized_);
  CHECK(rhs.is_normalized_);
  IntervalSet result(lhs.BitCount());
  std::list<Interval> lhs_intervals(lhs.Intervals().begin(),
                                    lhs.Intervals().end());
  std::list<Interval> rhs_intervals(rhs.Intervals().begin(),
                                    rhs.Intervals().end());
  auto left = lhs_intervals.begin();
  auto right = rhs_intervals.begin();
  // lhs/rhs_intervals should be sorted in increasing lexicographic order
  // (i.e.: compare on lower bound, then upper bound if lower bounds are equal)
  // at this point, since we CHECK that they are normalized.
  while ((left != lhs_intervals.end()) && (right != rhs_intervals.end())) {
    if (bits_ops::ULessThan(left->UpperBound(), right->UpperBound())) {
      if (std::optional<Interval> intersection =
              Interval::Intersect(*left, *right)) {
        result.AddInterval(*std::move(intersection));
        // The difference should only ever contain 0 or 1 interval.
        std::vector<Interval> difference = Interval::Difference(*right, *left);
        std::reverse(difference.begin(), difference.end());
        for (const Interval& remainder : difference) {
          right = rhs_intervals.insert(right, remainder);
        }
      }
      ++left;
      continue;
    }
    if (bits_ops::ULessThan(right->UpperBound(), left->UpperBound())) {
      if (std::optional<Interval> intersection =
              Interval::Intersect(*left, *right)) {
        result.AddInterval(*std::move(intersection));
        // The difference should only ever contain 0 or 1 interval.
        std::vector<Interval> difference = Interval::Difference(*left, *right);
        std::reverse(difference.begin(), difference.end());
        for (const Interval& remainder : difference) {
          left = lhs_intervals.insert(left, remainder);
        }
      }
      ++right;
      continue;
    }
    if (bits_ops::UEqual(left->UpperBound(), right->UpperBound())) {
      if (std::optional<Interval> intersection =
              Interval::Intersect(*left, *right)) {
        result.AddInterval(*std::move(intersection));
      }
      ++left;
      ++right;
      continue;
    }
  }
  result.Normalize();
  return result;
}

/* static */ IntervalSet IntervalSet::Of(absl::Span<Interval const> intervals) {
  IntervalSet result(intervals.front().BitCount());
  for (const Interval& a : intervals) {
    result.AddInterval(a);
  }
  result.Normalize();
  return result;
}

IntervalSet IntervalSet::Complement(const IntervalSet& set) {
  if (set.IsEmpty()) {
    return IntervalSet::Maximal(set.BitCount());
  }

  IntervalSet result(set.BitCount());
  Bits last_taken = Bits::AllOnes(set.BitCount());
  for (const Interval& interval : set.Intervals()) {
    // We rely on the fact that `set` is normalized if `Intervals()` was safe to
    // call, and that therefore:
    //   1. the intervals are sorted in increasing order by lower bound, and
    //   2. the intervals are not abutting or overlapping.
    Bits last_free = bits_ops::Decrement(interval.LowerBound());
    if (last_free != last_taken) {
      result.AddInterval(Interval::LeftOpen(last_taken, last_free));
    }
    last_taken = interval.UpperBound();
  }
  if (!last_taken.IsAllOnes()) {
    result.AddInterval(
        Interval::LeftOpen(last_taken, Bits::AllOnes(set.BitCount())));
  }
  result.Normalize();
  return result;
}

std::optional<int64_t> IntervalSet::Size() const {
  CHECK(is_normalized_);
  int64_t total_size = 0;
  for (const Interval& interval : intervals_) {
    std::optional<int64_t> size = interval.Size();
    if (!size.has_value()) {
      return std::nullopt;
    }
    static_assert(__has_builtin(__builtin_add_overflow));
    if (__builtin_add_overflow(total_size, *size, &total_size)) {
      return std::nullopt;
    }
  }
  return total_size;
}

std::optional<Bits> IntervalSet::Index(const Bits& index) const {
  CHECK(is_normalized_);
  CHECK_EQ(index.bit_count(), BitCount());
  Bits so_far = bits_ops::ZeroExtend(index, BitCount() + 1);
  std::optional<Interval> to_index;
  for (const Interval& interval : Intervals()) {
    Bits size = interval.SizeBits();
    if (bits_ops::ULessThan(so_far, size)) {
      to_index = interval;
      break;
    }
    so_far = bits_ops::Sub(so_far, size);
  }
  if (!to_index.has_value()) {
    return std::nullopt;
  }
  Bits result = bits_ops::Add(
      so_far,
      bits_ops::ZeroExtend(to_index.value().LowerBound(), BitCount() + 1));
  CHECK(!result.msb());
  return result.Slice(0, BitCount());
}

bool IntervalSet::IsTrueWhenMaskWith(const Bits& value) const {
  CHECK_EQ(value.bit_count(), BitCount());
  for (const Interval& interval : intervals_) {
    if (interval.IsTrueWhenAndWith(value)) {
      return true;
    }
  }
  return false;
}

bool IntervalSet::Covers(const Bits& bits) const {
  CHECK_EQ(bits.bit_count(), BitCount());
  for (const Interval& interval : intervals_) {
    if (interval.Covers(bits)) {
      return true;
    }
  }
  return false;
}

bool IntervalSet::IsPrecise() const {
  CHECK_GE(bit_count_, 0);
  std::optional<Interval> precisely;
  if (intervals_.empty()) {
    // A valueless interval is not precise. This can only happen if some
    // analysis hits a contradiction.
    return false;
  }
  for (const Interval& interval : intervals_) {
    if (precisely.has_value() && !(precisely.value() == interval)) {
      return false;
    }
    if (!interval.IsPrecise()) {
      return false;
    }
    precisely = interval;
  }
  return true;
}

std::optional<Bits> IntervalSet::GetPreciseValue() const {
  CHECK(is_normalized_);
  if (!IsPrecise()) {
    return std::nullopt;
  }
  CHECK_EQ(intervals_.size(), 1);
  return intervals_.front().GetPreciseValue();
}

bool IntervalSet::IsMaximal() const {
  CHECK(is_normalized_);
  CHECK_GE(bit_count_, 0);
  for (const Interval& interval : intervals_) {
    if (interval.IsMaximal()) {
      return true;
    }
  }
  return false;
}

bool IntervalSet::IsNormalized() const { return is_normalized_; }

absl::Status IntervalSet::CheckIsNormalizedForTesting() const {
  XLS_RET_CHECK(is_normalized_) << "Marked as not-normalized";
  XLS_RET_CHECK(absl::c_is_sorted(intervals_)) << "Intervals are not sorted";
  XLS_RET_CHECK(absl::c_none_of(intervals_, [](const Interval& i) {
    return i.IsImproper();
  })) << "Improper intervals!";
  XLS_RET_CHECK(absl::c_all_of(intervals_, [&](const Interval& i) {
    return i.BitCount() == bit_count_;
  })) << "incorrect bitcount!";
  for (const auto& vs : iter::sliding_window(intervals_, 2)) {
    XLS_RET_CHECK(!Interval::Overlaps(vs[0], vs[1]))
        << "Adjacent overlaping intervals " << vs[0] << " and " << vs[1];
    XLS_RET_CHECK(!Interval::Abuts(vs[0], vs[1]))
        << "Adjacent uncombined intervals " << vs[0] << " and " << vs[1];
  }
  return absl::OkStatus();
}

bool IntervalSet::IsEmpty() const {
  CHECK(IsNormalized());
  return intervals_.empty();
}

std::string IntervalSet::ToString() const {
  CHECK_GE(bit_count_, 0);
  std::vector<std::string> strings;
  strings.reserve(intervals_.size());
  for (const auto& interval : intervals_) {
    strings.push_back(interval.ToString());
  }
  return absl::StrFormat("[%s]", absl::StrJoin(strings, ", "));
}

xabsl::iterator_range<IntervalSet::SignedIntervalIterator>
IntervalSet::SignedIntervals() const {
  return {
      IntervalSet::SignedIntervalIterator(Intervals().cbegin(),
                                          Intervals().cend(), false),
      IntervalSet::SignedIntervalIterator(Intervals().cend(),
                                          Intervals().cend(), true),
  };
}

const Interval& IntervalSet::SignedIntervalIterator::operator*() const {
  const Interval& raw = *cur_;
  // TODO(allight): We could avoid building a bits object here. Probably not
  // worth it though.
  if (!InSplit()) {
    return raw;
  }
  // This interval needs to be split.
  // NB We clear the sign_swap_interval on every ++ so no need to check the
  // value.
  if (!sign_swap_interval_) {
    if (in_negatives_) {
      sign_swap_interval_ =
          Interval(Bits::MinSigned(raw.BitCount()), raw.UpperBound());
    } else {
      sign_swap_interval_ =
          Interval(raw.LowerBound(), Bits::MaxSigned(raw.BitCount()));
    }
  }
  return *sign_swap_interval_;
}

IntervalSet::SignedIntervalIterator&
IntervalSet::SignedIntervalIterator::operator++() {
  sign_swap_interval_.reset();
  auto raw = *cur_;

  if (InSplit() && !in_negatives_) {
    in_negatives_ = true;
    return *this;
  }
  ++cur_;
  return *this;
}

bool IntervalSet::SignedIntervalIterator::InSplit() const {
  if (cur_ == end_) {
    return false;
  }
  return cur_->Covers(Bits::MinSigned(cur_->BitCount())) &&
         cur_->Covers(Bits::MaxSigned(cur_->BitCount()));
}

}  // namespace xls
