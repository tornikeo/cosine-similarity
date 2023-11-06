#include <istream>
#include <fstream>
#include <iostream>
#include <iterator>
#include <utility>
#include <unordered_set>

#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>

using namespace std;
using namespace xt::placeholders;

template<class Stream, class Vector, class Begin = decltype(std::begin(std::declval<Vector>()))>
inline Stream& operator<<(Stream& stream, const Vector& vect) {
	const char* dlm = "";
	for(const auto& i : vect) { stream << dlm << i; dlm = ", "; }
	return stream;
}

xt::xarray<float> load(string filename) {
    cout << "Opening " << filename << " file " << endl;
    std::ifstream in_file;
    in_file.open(filename);
    cout << "Loading file in memory" << endl;
    auto references = xt::load_npy<float>(in_file);
    cout << "Data file loaded. It has " << references.shape() << " shape" << endl;
    in_file.close();
    return references;
}

int main()
{
    auto all_rmzs = load("references_mz.npy");
    auto all_qmzs = load("queries_mz.npy");
    auto all_rints = load("references_int.npy");
    auto all_qints = load("queries_int.npy");

    const auto R = all_rmzs.shape()[0];
    const auto M = all_rmzs.shape()[1];
    const auto Q = all_qmzs.shape()[0];
    const auto N = all_qmzs.shape()[1];
    const auto tolerance = 0.1f;
    const auto shift = 0.0f;
    const auto mz_power = 0.0f;
    const auto intensity_power = 1.0f;
    const auto T = static_cast<int>( max(M, N) );

    auto results_inds = vector< tuple< size_t, size_t > >(); 
    auto results_scores = vector< float >(); 

    // cout << vector<unsigned long int>({R, M, Q, N}) << endl;
    for (size_t r = 0; r < R; r++)
    {
        for (size_t q = 0; q < Q; q++)
        {
            auto rmzs = xt::row(all_rmzs, r);
            auto rints = xt::row(all_rints, r);
            
            auto qmzs = xt::row(all_qmzs, q);
            auto qints = xt::row(all_qints, q);
            
            size_t lowest_idx = 0;

            // find_matches
            auto matches = vector< pair<size_t, size_t> >();
            for (size_t peak1_idx = 0; peak1_idx < M; peak1_idx++)
            {
                auto mz = rmzs(peak1_idx);
                auto low_bound = mz - tolerance;
                auto high_bound = mz + tolerance;
                for (size_t peak2_idx = lowest_idx; peak2_idx < N; peak2_idx++)
                {
                    auto mz2 = qmzs(peak2_idx) + shift;
                    if (mz2 > high_bound) {
                        break;
                    }
                    if (mz2 < low_bound) {
                        lowest_idx = peak2_idx;
                    } else {
                        matches.push_back(pair(peak1_idx, peak2_idx));
                    }
                }
            }
            

            if (matches.empty()) {
                break;
            }

            auto idx1 = vector<size_t>();
            auto idx2 = vector<size_t>();

            for (pair i : matches)
            {
                idx1.push_back(i.first);
                idx2.push_back(i.second);
            }
            
            auto matching_pairs = vector< tuple<size_t, size_t, float> >();
            
            for (size_t i = 0; i < idx1.size(); i++)
            {
                auto idx = matches[i].first;
                auto idx2 = matches[i].second;

                auto spec1_mz = qmzs(idx);
                auto spec1_int = qints(idx);

                auto spec2_mz = rmzs(idx2);
                auto spec2_int = rints(idx2);

                auto power_prod_spec1 = pow(spec1_mz, mz_power) * pow(spec1_int, intensity_power);
                auto power_prod_spec2 = pow(spec2_mz, mz_power) * pow(spec2_int, intensity_power);

                matching_pairs.push_back(make_tuple(idx, idx2, power_prod_spec1 * power_prod_spec2));


            }

            auto score = 0.0f;
            auto used_matches = 0ul;
            // if (S.count(x)) { /* code */ }
            auto used1 = std::unordered_set<size_t>();
            auto used2 = std::unordered_set<size_t>();

            for (size_t i = 0; i < matching_pairs.size(); i++)
            {
                auto [idx, idx2, matching_score] = matching_pairs[i];
                if ( used1.find(idx) == used1.end() && used2.find(idx2) == used2.end() ) {
                        score += matching_score;
                        used1.insert(idx);
                        used2.insert(idx2);
                        used_matches++;
                }
            }

            // cout << score << endl;
            auto spec1_power = xt::pow(qmzs, mz_power) * xt::pow(qints, intensity_power);
            auto spec2_power = xt::pow(rmzs, mz_power) * xt::pow(rints, intensity_power);
            auto score_norm = xt::pow(xt::sum(xt::pow(spec1_power, 2.0)), 0.5) *
                              xt::pow(xt::sum(xt::pow(spec2_power, 2.0)), 0.5);
            // auto score_norm = xt::sum(spec1_power) * xt::sum(spec2_power);
            // auto score_norm = 0.0f;
            // for(auto it=spec1_power.begin(); it!=spec1_power.end(); ++it)
            // {
            //     score_norm += *it;
            // }
            
            // auto score_norm = xt::sum(spec1_power) * xt::sum(spec2_power);
            score = score / score_norm();
            results_inds.push_back(
                make_tuple(r, q)
            );
            results_scores.push_back(
                score
            );
            // if (!matching_pairs.empty()) {
            // }

            // cout << "Matches has " << matches.size() << " elements" << endl;
            // if (!matches.empty()) {
            //     cout << "Matches looks like " << matches[0].first << "," << matches[0].second << endl;
            //     // break;
            // }
            // cout << rmz.shape() << ", " << qmz.shape() << endl;
            // auto qmz = xt::row(qmzs, q, xt::all());
        }
    }



    
    // auto [Q, N] = qmz.shape().data();

    // cout << "Opening queries.npy file " << endl;
    // in_file.open("queries.npy");
    // cout << "Loading file in memory" << endl;
    // auto queries = xt::load_npy<float>(in_file);
    // cout << "Data file loaded. It has " << queries.shape() << " shape" << endl;
    // in_file.close();

    // auto ref_mz = xt::xview(references, xt::all(), xt::all(), 0);

    // for (size_t ref_id = 0; ref_id < references.shape()[0]; ref_id++)
    // {
    //     for (size_t q_id = 0; q_id < queries.shape()[0]; q_id++)
    //     {
    //         auto ref = references(ref_id);
    //         /* code */
    //     }
    // }

    auto xarr_results_scores = xt::adapt(results_scores, {results_scores.size(),1ul});
    xt::dump_npy("scores.npy", xarr_results_scores);

    auto xarr_results_inds = xt::adapt(results_inds, {results_inds.size(),1ul});
    xt::dump_npy("indices.npy", xarr_results_inds);
    
    // xt::xarray<float> a = {{1,2,3,4}, {5,6,7,8}};
    // xt::dump_npy("out.npy", a);

    return 0;
}
